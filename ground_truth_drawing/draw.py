#!/usr/bin/env python2
"""
    Simple interface for drawing the ground truth versions of shadow
    images.

    Shows the original image with the ground truth overlaid.

    - Holding down left click paints onto the ground truth layer.
    - Clicking middle mouse button switches the class you are currently
      painting with (e.g. switches from 'penumbra' to 'shadow' mode).
    - Holding down right-click erases things.
    - Scrolling up/down makes the painting radius larger/smaller.
    - Holding down shift whilst scrolling up/down zooms the image in/out.
    - Holding down shift + left click whilst moving mouse pans the image.
    - Holding down space hides the image (shows only ground truth).
    - Holding down return key hides the ground truth (shows only image).
    - Holding down x/y locks the cursor on that axis (for straight lines).
    - Left/Right keys: Switch to the previous/next image in the set.
    - 's' key saves the ground truth to the output directory.
    - 'f' key flood-fills an area with your currently selected class.
    - Holding down 'e' key shows detected edges in input image, locks
      cursor to nearest edge.

    If previous ground truth images exist, the program will attempt to
    load them when displaying the input images.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import argparse
import pygame
import numpy
import glob
import cv2
import sys
import os


# Pixel classes of the ground truth images.
CLASSES = {
    'shadow': 0,
    'penumbra': 25,
    'object': 153,
    'unknown': 250,
    'background': 255
}
DEFAULT_CLASS = 'background'
zoom = False
image = None
ground_truth = None
pan = (0, 0)
output_fmt = '.png'
output_dir = os.getcwd()
images = []
ground_truths = {}
image_idx = 0
clear = False
ground = True
line = False
alpha = 128
flood = False
axis = None
edges = None
show_edges = False


def class2rgb(pixel_class=DEFAULT_CLASS, alpha=255):
    """Takes the name of a pixel class and returns a pygame.Color RGB
       object."""
    return pygame.Color(*tuple([CLASSES[pixel_class]] * 3 + [alpha]))


class Cursor:
    """Handles storing of data for, and drawing of, a mouse cursor.
       Used for both the program's graphical cursor, and also for
       painting directly onto ground truth images.
    """
    def __init__(self):
        self.x, self.y = (0, 0)
        self.switch_class(DEFAULT_CLASS)
        self.old_class = DEFAULT_CLASS
        self.radius = 16
        self.active = False

    def draw(self, surface, pan=(0, 0)):
        """Paint cursor onto a specified surface. If the cursor is
           active, paints a filled circle. Otherwise, paints a circle
           with an outline.

           Arguments:
           surface: pygame.Surface onto which to draw this cursor.
           pan: If the above surface has been panned, use this tuple to
                specify how much it has been panned by, to offset the
                cursor when drawn to it.

           Class properties affect how it's drawn:
           cursor.active: if True, paint filled circle,
                          if False, paint ring
           cursor.radius: controls the radius of the circle that's drawn
           cursor.current_class_str: Name of the ground truth class that
                                     we're getting the cursor's colour
                                     from (e.g. 'shadow' or 'object').

           Returns: Nothing. Operates directly on the surface passed
                    to it.
        """
        if self.active is False:
            self.lastx, self.lasty = (None, None)
            pygame.draw.circle(
                surface,
                pygame.Color(255 - self.current_class,
                             (255 - self.current_class) * 2 % 255,
                             (255 - self.current_class) * 3 % 255),
                (self.x, self.y), self.radius + 1, 3
            )
            pygame.draw.circle(
                surface,
                class2rgb(self.current_class_str),
                (self.x, self.y), self.radius, 1
            )

        else:
            color = class2rgb(self.current_class_str)
            if (self.lastx, self.lasty) != (None, None):
                pygame.draw.line(
                    surface, color,
                    (self.lastx - pan[0], self.lasty - pan[1]),
                    (self.x - pan[0], self.y - pan[1]),
                    self.radius * 2
                )
            pygame.draw.circle(
                surface,
                class2rgb(self.current_class_str),
                (self.x - pan[0], self.y - pan[1]), self.radius, 0
            )
            self.lastx, self.lasty = (self.x, self.y)

    def switch_class(self, new_class=None):
        """Selects the next class in global list CLASSES, makes this
           cursor use that class."""
        if new_class is None:
            new_class_idx = (self.current_class_idx + 1) % len(CLASSES)
            new_class = list(CLASSES.keys())[new_class_idx]

        self.current_class_str = new_class
        self.current_class_idx = list(CLASSES.keys()).index(new_class)
        self.current_class = CLASSES[new_class]
        pygame.display.set_caption('mode: ' + self.current_class_str)

    def change_radius(self, radius):
        """Change the cursor's radius, as long as it's larger than 2.
           'radius' argument is a delta change to the cursor's actual
           current radius.
        """
        self.radius += radius
        if self.radius < 2:
            self.radius = 2

    def follow_closest_edge(self, edges, pos, pan=(0, 0)):
        """Snap cursor X,Y coordinates to closest white pixel + cursor
           radius."""
        edges_size = edges.get_size()
        black = pygame.Color(0x000000)
        surface_pos = (pos[0] + pan[0], pos[1] + pan[1])
        close = []
        half_radius = int(self.radius / 2)
        for x in range(-half_radius, half_radius + 1):
            for y in range(-half_radius, half_radius + 1):
                corner = (surface_pos[0] + x, surface_pos[1] + y)
                if corner[0] < 0 or corner[1] < 0:
                    continue
                if corner[0] >= edges_size[0] or corner[1] >= edges_size[1]:
                    continue
                if edges.get_at(corner) != black:
                    close.append((half_radius + x, half_radius + y))
        if len(close) > 0:
            self.x = pos[0] + (min(close)[0] - half_radius)
            self.y = pos[1] + (min(close)[1] - half_radius)
        else:
            self.x = pos[0]
            self.y = pos[1]


cursor = Cursor()


def flood_fill(surface, seed_point, color, pan=(0, 0)):
    """Flood fills a pygame surface, starting off at specific seed point.
       Returns the original surface with that area filled in.

       Thanks to wonderfully concise example of (non-recursive)
       flood-fill algorithm in Python:
       http://stackoverflow.com/a/11747550
    """
    seed_point = (seed_point[0] - pan[0], seed_point[1] - pan[1])

    if seed_point[0] > surface.get_width()\
    or seed_point[1] > surface.get_height()\
    or seed_point[0] < 0 or seed_point[1] < 0:
        return surface

    to_fill = set()
    to_fill.add(seed_point)
    background = surface.get_at(seed_point)
    while len(to_fill) > 0:
        x, y = to_fill.pop()
        surface.set_at((x, y), color)
        for i in range(x - 1, x + 2):
            if i < 0 or i > surface.get_width() - 1:
                continue
            for j in range(y - 1, y + 2):
                if j < 0 or j > surface.get_height() - 1:
                    continue
                if color != background\
                and surface.get_at((i, j)) == background:
                    to_fill.add((i, j))

    return surface


def get_ground_truth(input_file, output_dir, output_fmt):
    """Given a path to an input file, the output directory and output
       format (filename suffix), returns what the ground truth file's
       path should be."""
    name = os.path.splitext(os.path.basename(input_file))[0]
    output = os.path.join(output_dir, '%s.%s' % (name, output_fmt))
    return output


def handle_event(event):
    # TODO: This function is far too complex. Bits of it are repeated and could
    # probably be moved into their own functions with some thought.
    """Handle pygame events."""
    if event.type == pygame.QUIT:
        sys.exit(0)

    elif event.type == pygame.KEYDOWN:
        if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
            global zoom
            zoom = True

        elif event.key == pygame.K_SPACE:
            global clear
            clear = True

        elif event.key in[pygame.K_LCTRL, pygame.K_RCTRL]:
            global line
            line = True

        elif event.key == pygame.K_RETURN:
            global ground
            ground = False

        elif event.key == pygame.K_x:
            global axis
            axis = 'x'
        elif event.key == pygame.K_y:
            global axis
            axis = 'y'

        elif event.key == pygame.K_e:
            global show_edges
            show_edges = True

    elif event.type == pygame.KEYUP:
        if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
            global zoom
            zoom = False

        elif event.key == pygame.K_s:
            global ground_truth, images, image_idx, output_dir, output_fmt,\
                   ground_truths
            output = get_ground_truth(
                images[image_idx], output_dir, output_fmt
            )
            ground_truths[images[image_idx]] = output
            pygame.image.save(ground_truth, output)
            print('Saved', output)

        elif event.key == pygame.K_LEFT:
            global ground_truth, image, images, image_idx, pan, ground_truths,\
                   edges
            if image_idx > 0:
                image_idx -= 1
                pan = (0, 0)
                if images[image_idx] in ground_truths.keys():
                    ground_truth = pygame.image.load(
                        ground_truths[images[image_idx]]
                    )
                    ground_truth.set_colorkey(class2rgb())
                else:
                    ground_truth.fill(class2rgb())
                image = pygame.image.load(images[image_idx])
                edges = cv_edge_detect(images[image_idx])

        elif event.key == pygame.K_RIGHT:
            global ground_truth, image, images, image_idx, pan, ground_truths
            if image_idx < len(images) - 1:
                image_idx += 1
                pan = (0, 0)
                if images[image_idx] in ground_truths.keys():
                    ground_truth = pygame.image.load(
                        ground_truths[images[image_idx]]
                    )
                    ground_truth.set_colorkey(class2rgb())
                else:
                    ground_truth.fill(class2rgb())
                image = pygame.image.load(images[image_idx])
                edges = cv_edge_detect(images[image_idx])

        elif event.key == pygame.K_SPACE:
            global clear
            clear = False

        elif event.key in [pygame.K_LCTRL, pygame.K_RCTRL]:
            global line
            line = False

        elif event.key == pygame.K_RETURN:
            global ground
            ground = True

        elif event.key == pygame.K_f:
            global ground_truth, cursor, pan
            ground_truth = flood_fill(
                ground_truth, (cursor.x, cursor.y),
                class2rgb(cursor.current_class_str),
                pan
            )

        elif event.key in [pygame.K_x, pygame.K_y]:
            global axis
            axis = None

        elif event.key == pygame.K_e:
            global show_edges
            show_edges = False

    elif event.type == pygame.MOUSEMOTION:
        global cursor, zoom, axis, show_edges, edges
        if zoom is True and cursor.active is True:
            global pan
            pan = (pan[0] + event.rel[0], pan[1] + event.rel[1])
        if show_edges is True:
            global pan
            cursor.follow_closest_edge(edges, event.pos, pan)
        else:
            if axis is not 'y':
                cursor.x = event.pos[0]
            if axis is not 'x':
                cursor.y = event.pos[1]

    elif event.type == pygame.MOUSEBUTTONDOWN:
        global cursor
        if event.button == 1:
            cursor.active = True

        elif event.button == 3:
            cursor.active = True
            cursor.old_class = cursor.current_class_str
            cursor.switch_class(DEFAULT_CLASS)

    elif event.type == pygame.MOUSEBUTTONUP:
        global cursor
        if event.button == 1:
            cursor.active = False

        elif event.button == 2:
            cursor.switch_class()

        elif event.button == 3:
            cursor.active = False
            cursor.switch_class(cursor.old_class)

        elif event.button == 4:
            global zoom
            if zoom is True:
                global image, ground_truth, pan, edges
                image_w, image_h = image.get_size()
                gt_w, gt_h = ground_truth.get_size()
                edges_w, edges_h = edges.get_size()
                image = pygame.transform.scale(
                    image, (image_w * 2, image_h * 2)
                )
                ground_truth = pygame.transform.scale(
                    ground_truth, (gt_w * 2, gt_h * 2)
                )
                edges = pygame.transform.scale(
                    edges, (edges_w * 2, edges_h * 2)
                )
                pan = (pan[0] * 2, pan[1] * 2)

            else:
                cursor.change_radius(1)

        elif event.button == 5:
            global zoom
            if zoom is True:
                global image, ground_truth, pan, edges
                image_w, image_h = image.get_size()
                gt_w, gt_h = ground_truth.get_size()
                edges_w, edges_h = edges.get_size()
                image = pygame.transform.scale(
                    image, (int(image_w / 2), int(image_h / 2))
                )
                ground_truth = pygame.transform.scale(
                    ground_truth, (int(gt_w / 2), int(gt_h / 2))
                )
                edges = pygame.transform.scale(
                    edges, (int(edges_w / 2), int(edges_h / 2))
                )
                pan = (int(pan[0] / 2), int(pan[1] / 2))

            else:
                cursor.change_radius(-1)

    elif event.type == pygame.VIDEORESIZE:
        global window
        window = pygame.display.set_mode(event.size, pygame.RESIZABLE)


def find_input_images(working_dir=os.getcwd(), image_format='jpg'):
    """Find the input images. Defaults to looking for all .jpg files in
       current working directory."""
    images = glob.glob(os.path.join(working_dir, '*.%s' % image_format))
    if images is None:
        return []
    return images


def cv_edge_detect(image):
    """Use OpenCV2 to perform Canny Edge detection on an input image.

       Returns as a new pygame surface."""
    cv_surface = cv2.imread(image, cv2.CV_LOAD_IMAGE_COLOR)
    cv_surface = cv2.cvtColor(cv_surface, cv2.COLOR_BGR2GRAY)
    cv_surface = cv2.fastNlMeansDenoising(cv_surface)
    edges = cv2.Canny(cv_surface, 100, 100, apertureSize=5)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    new_surface = pygame.image.frombuffer(
        edges.tostring(), edges.shape[1::-1], 'RGB'
    )
    new_surface.set_colorkey(0x000000)
    return new_surface


def parse_program_arguments():
    """Parses command line arguments and returns a Namespace."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-o', '--output-dir', default=os.getcwd())
    parser.add_argument('-i', '--input-images', nargs='+', type=str,
                        default=find_input_images())
    parser.add_argument('-f', '--output-format', default='png')
    return parser.parse_args()


def main():
    args = parse_program_arguments()
    if args.input_images is None or len(args.input_images) == 0:
        sys.exit('No input images were found.')

    global window, cursor, image, ground_truth, images, output_fmt, output_dir
    global clear, line, ground, ground_truths, edges, show_edges

    output_dir = args.output_dir
    output_fmt = args.output_format
    images = args.input_images

    # Look for existing ground truth images for input images.
    for image_file in images:
        ground_truth_file = get_ground_truth(
            image_file, output_dir, output_fmt
        )
        if os.path.isfile(ground_truth_file):
            ground_truths[image_file] = ground_truth_file

    # Load the first input image and its ground truth.
    image = pygame.image.load(images[0])
    edges = cv_edge_detect(images[0])
    edges.set_colorkey(0x000000)
    if images[0] in ground_truths.keys():
        ground_truth = pygame.image.load(ground_truths[images[0]])
    else:
        ground_truth = pygame.Surface(image.get_size(), pygame.SRCCOLORKEY)
        ground_truth.fill(class2rgb())
    ground_truth.set_colorkey(class2rgb())

    # Open pygame window, set to the same size as the input image.
    window = pygame.display.set_mode(
        image.get_size(), pygame.RESIZABLE | pygame.SRCALPHA
    )
    cursor = Cursor()
    pygame.mouse.set_visible(False)

    while True:
        window.fill(class2rgb())  # clear previous frame from window
        ground_truth.set_alpha(alpha)  # make ground truth transparent
        handle_event(pygame.event.wait())  # wait for an event, handle it
        for event in pygame.event.get():  # handle any other queued events
            handle_event(event)

        # If left mouse button is down, paint onto the ground truth surface
        if cursor.active is True and zoom is False:
            cursor.draw(ground_truth, pan)

        # Draw the surfaces onto the window.
        # Image is first to be blitted, then ground truth, then the cursor.
        if clear is False:
            window.blit(image, pan)
        if ground is True:
            window.blit(ground_truth, pan)
        if show_edges is True:
            window.blit(edges, pan)
        if cursor.active is False:
            cursor.draw(window)

        pygame.display.flip()


if __name__ == '__main__':
    main()

# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
