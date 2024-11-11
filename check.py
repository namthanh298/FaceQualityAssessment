
import cv2
import numpy as np
from scipy.stats import kurtosis


def calculate_blur_fft(image_path):
    """
    Tính toán độ blur của ảnh thông qua miền tần số.

    Args:
      image_path: Đường dẫn đến ảnh.

    Returns:
      Độ blur của ảnh (giá trị càng thấp càng mờ).
    """
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Biến đổi Fourier
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Tính toán năng lượng phổ ở vùng tần số cao và thấp
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    high_freq_energy = np.sum(magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10])
    low_freq_energy = np.sum(magnitude_spectrum) - high_freq_energy

    # Tính toán tỷ lệ năng lượng
    blur_score = 1 - high_freq_energy / low_freq_energy

    return blur_score

def skin_content_measure(image_path):
  """
  Hàm tính toán tỷ lệ màu da trong ảnh.

  Args:
    image_path: Đường dẫn đến ảnh.

  Returns:
    Tỷ lệ pixel có màu da (trong khoảng 0 đến 1).
  """
  # Đọc ảnh
  img = cv2.imread(image_path)

  # Chuyển đổi sang không gian màu HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Tách các kênh Hue, Saturation, Value
  h, s, v = cv2.split(hsv)

  # Điều kiện về Hue và Saturation
  lower_skin = np.array([0, 5, 0], dtype=np.uint8)
  upper_skin = np.array([30, 95, 255], dtype=np.uint8)

  # Tạo mask cho các pixel thỏa mãn điều kiện
  mask = cv2.inRange(hsv, lower_skin, upper_skin)

  # Tính toán tỷ lệ pixel có màu da
  skin_pixels = cv2.countNonZero(mask)
  total_pixels = img.shape[0] * img.shape[1]
  skin_ratio = skin_pixels / total_pixels

  return skin_ratio

def calculate_blur_kurtosis_fft(image_path):
    """
    Tính toán độ blur của ảnh bằng kurtosis trên miền tần số.

    Args:
      image_path: Đường dẫn đến ảnh.

    Returns:
      Kurtosis của phổ tần số (giá trị càng cao càng nét).
    """
    # Đọc ảnh và chuyển sang ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Biến đổi Fourier 2D
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Tính kurtosis
    kurtosis_value = kurtosis(magnitude_spectrum, fisher=False)

    return kurtosis_value