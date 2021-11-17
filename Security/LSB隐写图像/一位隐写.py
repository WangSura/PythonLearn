# https://blog.csdn.net/qq1437715969/article/details/103617581
from PIL import Image as im
import re

replace_reg = re.compile(r'[1|0]$')

# 替换最后一位的数据，source是被替换数据，target是目标数据，就是batarget放到source最后一位


def repLstBit(source, target):
    return replace_reg.sub(target, source)
    # 运行结果：'123X'
    print(repLstBit("111110", "1"))

    # 字符串转换二进制，不够八位的话补齐8位


def encode(s):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in s)

    # 切割从图像中收集到的数据，就是把载密图像的对应最后一位提取出来之后需要进行切割


def cut_text(text, lenth):
    textArr = re.findall('.{'+str(lenth)+'}', text)
    tempStr = text[(len(textArr) * lenth):]
    if len(tempStr) != 0:
        textArr.append(text[(len(textArr)*lenth):])
    return textArr

    # 二进制转换成字符串，看上面切割方法的注释即可理解该方法存在的意义


def decode(s):
    bitArr = cut_text(s, 8)
    return "".join(chr(int(i, 2)) for i in bitArr)


    # 读取宿主图像和要写入的信息生成载密图像。
if __name__ == '__main__':
    img = im.open("D:\program\pythonPractice\Security\LSB隐写图像\dove.png")
    width = img.size[0]
    height = img.size[1]
    hideInfo = "Hello ImageSteg"
    hideBitArr = encode(hideInfo)
    count = 0
    bitInfoLen = len(hideBitArr)

    print(hideBitArr)
    for i in range(width):
        for j in range(height):
            if count == bitInfoLen:
                break
            pixel = img.getpixel((i, j))
            print(pixel[0])
            sourceBit = bin(pixel[0])[2:]
            print(sourceBit)
            rspBit = int(repLstBit(sourceBit, hideBitArr[count]), 2)
            count += 1
            img.putpixel((i, j), (rspBit, rspBit, rspBit))
    img.save("D:\program\pythonPractice\Security\LSB隐写图像\dove2.png")
