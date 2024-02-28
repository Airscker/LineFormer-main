import infer
import cv2
import line_utils
import matplotlib.pyplot as plt

img_path = "demo/NMR 2.jpg"
img = cv2.imread(img_path)  # BGR format

CKPT = "iter_3000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
# print(infer.model)
line_dataseries, masks = infer.get_dataseries(img,
                                              to_clean=False,
                                              return_masks=True)
print('mask number:',len(masks))
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(masks[0])
# plt.show()
# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
cv2.imwrite(img_path.replace('.', '_res.'), img)
