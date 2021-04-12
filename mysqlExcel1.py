import pymysql       ##PyMySQL是在 Python3.x 版本中用于连接 MySQL 服务器的一个库
import pandas as pd  ##Pandas是Python的一个数据分析包 导入panda命名为pd
from sqlalchemy import create_engine ## 导入引擎

file = r'd:/database/mysql/bin/test.xlsx'    ## 文件
df = pd.read_excel(file) ## 读文件

 ## 连接数据库
 ##都要小写什么鬼
engine = create_engine("mysql+mysqlconnector://test:890765@localhost/test") ## ****代表你的数据库密码
df.to_sql('student',con=engine,if_exists='replace',index=False)  ##导入数据库，如果存在就替换
##if_exists='replace' 如果存在就替换，那追加呢？

##if_exists='append'