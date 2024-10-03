import random
import numpy as np
import cv2

class Degrade():
    def __init__(self,name):
        self.name = name

class fwAdder(Degrade):
    def __init__(self):
        super(fwAdder,self).__init__("origin")
    def __call__(self,sample):
        return sample

class BlurAdder(Degrade):
    def __init__(self,mode="random",sigma=2):
        super(BlurAdder,self).__init__("blur")
        self.mode = mode
        self.sigma = sigma
        self.sigma_range = np.arange(2,3.1,0.1)
        self.kernel_size = 15
    def blurring(self,image):  # gamma函数处理
        # 随机选择一个数值
        kernel = cv2.getGaussianKernel(self.kernel_size, self.sigma)
        kernel = np.outer(kernel, kernel.transpose())
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image

    def __call__(self,img):
        img = np.clip(np.asarray(img)*255,0,255)

        if self.mode == "random":
            self.sigma = random.choice(self.sigma_range)

        x = self.blurring(img)
        # x = Image.fromarray(x)

        return x/255

class LightAdder(Degrade):
    def __init__(self,mode="random",gamma=2.):
        super(LightAdder,self).__init__("dark")
        self.mode = mode
        self.gamma = gamma
        self.a = 0.95
        self.b = 0.6
    def gamma_trans(self,img):  # gamma函数处理

        gamma_table = [self.b*(self.a*np.power(x / 255.0, self.gamma)) * 255.0 for x in range(256)]  # 建立映射表
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
        return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

    def __call__(self,img):
        img = np.clip(np.asarray(img)*255,0,255)

        if self.mode == "random":
            self.gamma = random.uniform(2,3.5) # 公式计算gamma
            self.a=random.uniform(0.9,1)
            self.b=random.uniform(0.5,1)

        x = self.gamma_trans(img)
        # x = Image.fromarray(x)
        return x/255.0

class NoiseAdder(Degrade):
    def __init__(self,mode="random",var=50.):
        super(NoiseAdder,self).__init__("noise")         
        self.mode = mode
        self.var = var

    def add_gaussian_noise(self,image):
        # 生成均值为0，方差为指定值的高斯噪声
        noise = np.random.normal(0, self.var, image.shape)

        # 将噪声添加到图像
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return noisy_image

    def __call__(self,img):
        img = np.clip(np.asarray(img)*255,0,255)

        if self.mode == "random":
            self.var = random.randint(0, 50)

        x = self.add_gaussian_noise(img)

        return x/255.0

class JpegAdder(Degrade):
    def __init__(self,mode="random",q=85.):
        super(JpegAdder,self).__init__("jpeg")   
        self.mode = mode
        self.quality = q

    def jpeg(self,image):
        # 执行JPEG压缩
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])

        # 解码压缩后的图像
        result_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        return result_image

    def __call__(self,img):
        img = np.clip(np.asarray(img)*255,0,255)

        if self.mode == "random":
            self.quality = random.randint(10, 90)

        x = self.jpeg(img)

        return x/255.0

class AugmentatorHub(Degrade):
    def __init__(self,adderList):
        super(AugmentatorHub, self).__init__("hub")
        self.adders = adderList
    def __call__(self, img):
        mode = random.randint(0, len(self.adders)-1)

        img = self.adders[mode](img)

        return img

def parse_degradations(augment_opt):
    augmentators = []
    if augment_opt['origin']:
        augmentators.append(fwAdder())
    if augment_opt['noise']:
        setting = augment_opt['noise']
        mode = setting.get('mode','random')
        para = setting.get('var', 25)
        augmentators.append(NoiseAdder(mode,para))
    if augment_opt['blur']:
        setting = augment_opt['blur']
        mode = setting.get('mode','random')
        para = setting.get('sigma', 2.0)
        augmentators.append(BlurAdder(mode,para))
    if augment_opt['jpeg']:
        setting = augment_opt['jpeg']
        mode = setting.get('mode','random')
        para = setting.get('q', 50)
        augmentators.append(JpegAdder(mode,para))
    if len(augmentators) == 0:
        augmentators.append(fwAdder())
        
    return augmentators