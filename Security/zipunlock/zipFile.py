
import zipfile


def extractFile(zFile, password):
    try:
        zFile.extractall(pwd=str.encode(password))
    # 如果成功返回密码
        return password
    except:
        return


def main():
    zFile = zipfile.ZipFile(
        "D:\program\pythonPractice\Security\zipunlock\q.zip", "r")
    # 打开我们的字典表

    f = open('D:\program\pythonPractice\Security\zipunlock\zidian.txt', 'w')
    for id in range(1000000):
        password = str(id).zfill(6)+'\n'
        f.write(password)
    f.close()

    passFile = open('D:\program\pythonPractice\Security\zipunlock\zidian.txt')
    for line in passFile.readlines():
        # 读取每一行数据（每一个密码)
        password = line.strip('\n')
        guess = extractFile(zFile, password)
        if (guess):
            print("=========code is："+password+"\n")
            exit(0)


if __name__ == '__main__':
    main()
