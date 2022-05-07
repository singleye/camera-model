import cv2
import math
import time
import numpy as np
from threading import Lock, Thread


class Object3D(object):
    def __init__(self):
        self._scene = None
        self._color = (0x00, 0x00, 0x00)

    def bind(self, scene):
        '''
        Bind the object to a scene
        '''
        self._scene = scene

    def refresh(self, camera):
        '''
        Refresh the object in the view of camera
        '''
        raise Exception('Not implemented!')


class Point(Object3D):
    '''
    3D space point
    '''

    def __init__(self, x, y, z, color=(0xFF, 0xFF, 0xFF), thickness=1):
        super().__init__()
        self._x, self._y, self._z = x, y, z
        self._color = color
        self._thickness = thickness

    def refresh(self, camera):
        v_c = camera.trans_to_cam(np.array([[self._x, self._y, self._z]]))
        proj_v = camera.project(v_c)
        camera.draw_point_2d(round(proj_v[0][0]), round(proj_v[0][1]), self._color, self._thickness)


class Line(Object3D):
    '''
    3D space line
    '''

    def __init__(self, start, end, color=(0xFF, 0xFF, 0xFF), thickness=1):
        super().__init__()
        self._start, self._end = np.array(start), np.array(end)
        self._color = color
        self._thickness = thickness

    def refresh(self, camera):
        start_c = camera.trans_to_cam(np.array([self._start]))
        start_p = camera.project(start_c)
        end_c = camera.trans_to_cam(np.array([self._end]))
        end_p = camera.project(end_c)
        camera.draw_line_2d(tuple(start_p[0].astype(int)),
                            tuple(end_p[0].astype(int)),
                            self._color, self._thickness)


class Box(Object3D):
    def __init__(self, center, size, color=(0xFF, 0xFF, 0xFF)):
        super().__init__()
        self._center = center
        self._size = size

    def refresh(self, camera):
        pass


class Canvas(object):
    '''
    A canvas object
    '''

    def __init__(self, width, height):
        self._width = width
        self._height = height

    def clean(self):
        pass


class Camera(object):
    def __init__(self, width, height, fps=30):
        # camera position in world frame
        self._x, self._y, self._z = 0, 0, 0
        self._pos = np.array([self._x, self._y, self._z])
        self._fps = fps

        # camera rotation
        self._roll, self._pitch, self._yaw = 0, 0, 0
        self.R = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self._distance = 0

        self._canvas_width, self._canvas_height = width,height
        self._canvas_shown = np.zeros((height, width, 3))
        self._canvas_hidden = np.zeros((height, width, 3))
        self._canvas_lock = Lock()

        # camera focus length in meter
        self._f = 1
        # scale factor: pixels/meter
        self._s = 800
        # NOTE:
        #   If you want output to be [u, v, f], then self.intrinsic[3][3] should be self._f,
        #   but normally we output [u, v, 1], then self.intrinsic[3][3] is 1
        self.intrinsic = np.array([[self._f*self._s,    0,                  self._canvas_width/2.0],
                                   [0,                  self._f*self._s,    self._canvas_height/2.0],
                                   [0,                  0,                  1]])

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, value):
        self._roll = value
        self.rotate(self._roll, self._pitch, self._yaw)

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        self.rotate(self._roll, self._pitch, self._yaw)

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self.rotate(self._roll, self._pitch, self._yaw)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value

    @property
    def focus(self):
        return self._f

    @focus.setter
    def focus(self, value):
        self._f = value
        self.intrinsic[0][0] = self._f * self._s
        self.intrinsic[1][1] = self._f * self._s

    def move(self, x, y, z):
        '''
        Move camera to new position
        '''
        self._x, self._y, self._z = x, y, z

    def rotate(self, roll, pitch, yaw):
        '''
        Rotate camera by roll, pitch, yaw
        '''
        rx, _ = cv2.Rodrigues((pitch, 0, 0))
        ry, _ = cv2.Rodrigues((0, yaw, 0))
        rz, _ = cv2.Rodrigues((0, 0, roll))
        self.R = np.dot(rz, np.dot(ry, rx))

    def trans_to_cam(self, v):
        '''
        Transform the world coordinate vertices to camera coordinate vertices
            v: vertices in world coordinate frame
        '''
        #vc = np.dot(self.R, v.T) - np.array([[self._x], [self._y], [self._z]])
        vc = np.dot(self.R, (v.T - np.array([[self._x], [self._y], [self._z]])))
        return vc.T

    def project(self, v):
        '''
        Project the vertices of camera coordinate to camera image plane coordinate
            v: vertices in camera coordinate frame
                u = width - f*(Y/X)
                v = height - f*(Z/X)
        '''
        #proj_v = [self._canvas_width, self._canvas_height] - self._f*v[:, 1:]/v[:, 0, np.newaxis]
        #proj_v = np.flip(proj_v, axis=1)

        Z = np.expand_dims(v[:, -1], axis=1)
        #proj_v = (self._f / Z) * np.dot(self.intrinsic, v.T)
        proj_v = np.dot(self.intrinsic, v.T) / Z
        proj_v = proj_v.T
        proj_v[:, 0:2] = [self._canvas_width, self._canvas_height] - proj_v[:, 0:2]
        return proj_v[:, :2]

    def draw_point_2d(self, x, y, color=(0xFF, 0xFF, 0xFF), thickness=1):
        '''
        Draw a point on the canvas
        '''
        with self._canvas_lock:
            #selector = v.astype(int)
            #self._canvas_hidden[y, x] = color
            cv2.circle(self._canvas_hidden, (int(x), int(y)), math.ceil(thickness/2.0), color)

    def draw_line_2d(self, start, end, color, thickness=1):
        '''
        Draw a line on the canvas
        '''
        with self._canvas_lock:
            cv2.line(self._canvas_hidden, start, end, color, thickness, cv2.LINE_AA)

    def render(self, v, color):
        with self._canvas_lock:
            selector = v.astype(int)
            self._canvas_hidden[selector[:, 0], selector[:, 1]] = color

    def clean_canvas(self):
        self._canvas_hidden = np.zeros((self._canvas_height, self._canvas_width, 3))

    def flush_canvas(self):
        with self._canvas_lock:
            self._canvas_shown = self._canvas_hidden
            self._canvas_hidden = None

    def play(self, name, show_fps=True):
        '''
        view the scene
        '''
        cv2.namedWindow(name)
        fps = self._fps
        info_toggle = True
        while True:
            frame_start = time.time()
            with self._canvas_lock:
                if info_toggle:
                    w_offset = 200
                    h_offset = 20
                    font_size = 0.3
                    font_thickness = 1
                    cv2.putText(self._canvas_shown,
                                'Camera',
                                (self._canvas_width-w_offset, h_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                (0xA0, 0xA0, 0xA0), font_thickness, cv2.LINE_AA)
                    h_offset += 10
                    cv2.putText(self._canvas_shown,
                                '  Position: x: %.1f, y: %.1f, z: %.1f' % (self._x, self._y, self._z),
                                (self._canvas_width-w_offset, h_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                (0xA0, 0xA0, 0xA0), font_thickness, cv2.LINE_AA)
                    h_offset += 10
                    cv2.putText(self._canvas_shown,
                                '  Rotation: R: %.2f, P: %.2f, Y: %.2f' % (self._roll, self._pitch, self._yaw),
                                (self._canvas_width-w_offset, h_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                (0xA0, 0xA0, 0xA0), font_thickness, cv2.LINE_AA)
                    h_offset += 10
                    cv2.putText(self._canvas_shown,
                                '  Focus: %.2f' % (self.focus),
                                (self._canvas_width-w_offset, h_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                (0xA0, 0xA0, 0xA0), font_thickness, cv2.LINE_AA)
                cv2.imshow(name, self._canvas_shown)

            if show_fps:
                win_name = '%s Size(%d,%d) %d fps' % (name, self._canvas_width, self._canvas_height, round(fps))
            else:
                win_name = name

            cv2.setWindowTitle(name, win_name)

            key = cv2.waitKey(1) & 0xFF
            # ESC
            if key == 0x1b:
                break
            # LEFT
            elif key == 0x51 or key == 0x02:
                self.yaw += np.pi/180
            # UP
            elif key == 0x52 or key == 0x00:
                self.pitch -= np.pi/180
            # RIGHT
            elif key == 0x53 or key == 0x03:
                self.yaw -= np.pi/180
            # DOWN
            elif key == 0x54 or key == 0x01:
                self.pitch += np.pi/180
            # ',/<'
            elif key == 0x2c:
                self.roll += np.pi/180
            # './>'
            elif key == 0x2e:
                self.roll -= np.pi/180
            # '-'
            elif key == ord('-'):
                new_f = self.focus - 0.1
                self.focus = max(new_f, 0.1)
            # '+'
            elif key == ord('='):
                new_f = self.focus + 0.1
                self.focus = min(new_f, 2000)
            elif key == ord('q'):
                self._z -= 0.1
            elif key == ord('e'):
                self._z += 0.1
            elif key == ord('w'):
                self._y += 0.1
            elif key == ord('s'):
                self._y -= 0.1
            elif key == ord('a'):
                self._x += 0.1
            elif key == ord('d'):
                self._x -= 0.1
            # info
            elif key == ord('i'):
                info_toggle = not info_toggle

            time_render = time.time() - frame_start
            frame_sleep = max(1.0/self._fps - time_render, 0)
            time.sleep(frame_sleep)
            fps = 1.0/(time.time() - frame_start)


class Scene(Thread):
    def __init__(self, name):
        Thread.__init__(self)
        self._name = name
        self._cam = None
        self._objects = []
        self._running = True

    def _refresh(self):
        '''
        Refresh the objects onto the canvas
        '''
        self._cam.clean_canvas()
        for obj in self._objects:
            obj.refresh(self._cam)
        self._cam.flush_canvas()

    def run(self):
        while self._running:
            self._refresh()

    def set_camera(self, camera):
        '''
        Set the viewing camera
        '''
        self._cam = camera

    def show(self):
        # start thread
        self.start()

        # start rendering
        self._cam.play(self._name)

        # TODO: quit logic
        self._running = False
        self.join()

    def draw_point_3d(self, x, y, z, color=(0xFF, 0xFF, 0xFF), thickness=1):
        '''
        Draw a point on the canvas
        '''
        pt = Point(x, y, z, color, thickness)
        pt.bind(self)
        self._objects.append(pt)

    def draw_line_3d(self, start, end, color=(0xFF, 0xFF, 0xFF), thickness=1):
        '''
        Draw a line on the canvas
            start: start position (tuple)
            end: end position (tuple)
        '''
        line = Line(start, end, color, thickness)
        line.bind(self)
        self._objects.append(line)


def test():
    scene = Scene('Hello world!')
    cam = Camera(800, 640)
    cam.move(0, 0, -10)
    scene.set_camera(cam)

    # add origin axis marker
    scene.draw_line_3d((0, 0, 0), (1, 0, 0), color=(0x00, 0x00, 0xFF), thickness=1)
    scene.draw_line_3d((0, 0, 0), (0, 1, 0), color=(0x00, 0xFF, 0x00), thickness=1)
    scene.draw_line_3d((0, 0, 0), (0, 0, 1), color=(0xFF, 0x00, 0x00), thickness=1)

    # add a grid
    length = 10
    grid = 10
    grid_x = -length/2.0
    grid_z = -length/2.0
    for i in range(0, grid+1):
        scene.draw_line_3d((grid_x, -1, grid_z), (grid_x, -1, grid_z+length), color=(0xA0, 0xA0, 0xA0), thickness=1)
        grid_x += 1.0*length/grid

    grid_x = -length/2.0
    grid_z = -length/2.0
    for i in range(0, grid+1):
        scene.draw_line_3d((grid_x, -1, grid_z), (grid_x+length, -1, grid_z), color=(0xA0, 0xA0, 0xA0), thickness=1)
        grid_z += 1.0*length/grid

    # add a point
    scene.draw_point_3d(0.1, -0.1, 1, color=(0x00, 0xFF, 0x00))
    scene.draw_point_3d(0, 0, 0, color=(0x00, 0xFF, 0x00))

    scene.show()

if __name__ == '__main__':
    test()
