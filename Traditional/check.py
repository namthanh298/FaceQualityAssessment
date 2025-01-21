
import cv2
import numpy as np
from scipy.stats import kurtosis, skew, entropy


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

def calculate_blur_fft(img):
    """
    Tính toán độ blur của ảnh thông qua miền tần số.

    Returns:
      Độ blur của ảnh (giá trị càng thấp càng mờ).
    """
    # Đọc ảnh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


def compare_with_blurred_version(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.boxFilter(img_gray, -1, (3,3))

    score = cv2.PSNR(img_gray, blurred)
    return score

def calculate_brightness_moments(img):
    """
    Tính toán các brightness moment (mean, variance, skewness, kurtosis) của ảnh
    và chuẩn hóa các giá trị về khoảng (0, 1).

    Returns:
        Một tuple chứa các giá trị đã chuẩn hóa của mean, variance, skewness, kurtosis.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tính toán mean
    mean = np.mean(gray)
    mean_norm = mean / 255

    # Tính toán variance
    variance = np.var(gray)
    variance_norm = variance / np.max(gray) ** 2

    # Tính toán skewness
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    skewness = skew(hist)[0]
    # skewness_norm = (skewness + 3) / 6  # Giả sử skewness nằm trong khoảng (-3, 3)

    # Tính toán kurtosis
    kurtosis_value = kurtosis(hist)[0]
    # kurtosis_norm = (kurtosis_value + 3) / 6  # Giả sử kurtosis nằm trong khoảng (-3, 3)

    return mean_norm, variance_norm, skewness, kurtosis_value

def calculate_contrast(img):
    """
    Tính toán các độ đo tương phản của ảnh, bao gồm RMS contrast và Michelson contrast
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # RMS contrast
    rms_contrast = img.std()
    # Michelson contrast
    lmin = np.min(gray)
    lmax = np.max(gray)
    michelson_contrast = (lmax-lmin) / (lmax+lmin)
    
    return rms_contrast, michelson_contrast

def calculate_sharpness(img):
    """
    Tính toán các độ đo cho độ sắc nét (hoặc độ mờ) của ảnh.
    """
    # Laplacian variance
    lap = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_var = lap.var()
    
    # Modified Laplacian
    Lx = cv2.filter2D(img, cv2.CV_32F, np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]))
    Ly = cv2.filter2D(img, cv2.CV_32F, np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]))
    modified_laplacian = (np.abs(Lx) + np.abs(Ly)).mean()
    
    # Laplacian energy
    laplacian_energy = np.square(lap).mean()
    
    # Tenengrad
    sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    tenengrad = cv2.magnitude(sx, sy).mean()

    return laplacian_var, modified_laplacian, laplacian_energy, tenengrad

def calculate_colorfulness(img):
    # split the image into its respective RGB components
	(B, G, R) = cv2.split(img.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def calculate_noise_metrics(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Calculate the noise by subtracting the blurred image from the original grayscale image
    noise = gray_image - blurred_image

    # Calculate the mean and standard deviation of the noise
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)

    return mean_noise, std_noise

def calculate_entropy(img):
    """
    Tính toán entropy của ảnh. Entropy là đại đo lường mức độ "rối loạn", "không chắc chắn" của thông tin trong ảnh.
    - Entropy cao: Ảnh chứa nhiều thông tin, các mức xám phân bố đều
    - Entropy thấp: Ảnh chứa ít thông tin, các mức xám tập trung vào một số giá trị nhất định
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256]) / gray.size
    # entropy = -np.sum(hist * np.log2(hist + 1e-10))
    entropy_value = entropy(img.flatten())
    return entropy_value

def skin_content_measure(img):
    """
    Tính toán tỷ lệ pixel có màu da (trong khoảng 0 đến 1).
    """    
    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Tách các kênh Hue, Saturation, Value
    h, s, v = cv2.split(hsv)
    
    # Điều kiện về Hue và Saturation
    lower_skin = np.array([0, 30, 0], dtype=np.uint8)
    upper_skin = np.array([30, 240, 255], dtype=np.uint8)
    
    # Tạo mask cho các pixel thỏa mãn điều kiện
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Tính toán tỷ lệ pixel có màu da
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    skin_ratio = skin_pixels / total_pixels
    
    return skin_ratio

def calculate_num_keypoints(img):
    # Số lượng điểm đặc trưng (SIFT)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(img, None)
    num_keypoints = len(keypoints)
    return num_keypoints
    
