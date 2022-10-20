import math
import cv2
import base64


def _round_aspect(number, key):
    return max(min(math.floor(number), math.ceil(number), key=key), 1)


def resize_keeping_aspect_ratio(pil_image, target_size, resample=None):
    x, y = target_size
    aspect = pil_image.size[0] / pil_image.size[1]
    if x / y >= aspect:
        x = _round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = _round_aspect(
            x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n)
        )
    target_size = (x, y)

    return pil_image.resize(target_size, resample), target_size


def image_to_base64(cv2_image, ext='png'):
    _, im_arr = cv2.imencode('.{}'.format(ext), cv2_image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode('ascii')
