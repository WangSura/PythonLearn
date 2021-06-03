import random as r
import pymysql 
first=('张','王','李','赵','金','艾','单','龚','钱','周','吴','郑','孔','曺','严','华','吕','徐','何')
middle=('芳','军','建','明','辉','芬','红','丽','功')
last=('明','芳','','民','敏','丽','辰','楷','龙','雪','凡','锋','芝','')
name=[]
passwd1=('1234','5678','147','258')

for i in range(101):   
    name1=r.choice(first)+r.choice(middle)+r.choice(last) #末尾有空格的名字
    name2=name1.rstrip()  #去掉末尾空格后的名字
    if name2 not in name: #名字存入列表中，且没有重名
        name.append(name2)
                
conn = pymysql.connect(host='127.0.0.1', port=3306, user='test', passwd='890765',db='shopsystem')
cur = conn.cursor() 

for i in range(len(name)):      #插入数据
    passwd=r.choice(passwd1)    #在密码列表中随机取一个
    #cur.execute("insert into member(name,passwd) values(%s,%s)",(name[i],passwd))#注意用法
    sql = "insert into member(id,tel,password,wx_token,im_token,open_id,status,created_at,updated_at) \
    values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    
    #cur.execute("insert into member(id,tel,password,wx_token,im_token,open_id,status,created_at,updated_at) \
    #values(%s,%s)",(name[i],passwd))#注意用法
    
    cur.execute(sql,(name[i],passwd))#注意用法
cur.execute('select * from menber') #查询数据
for s in cur.fetchall():
    print(s)
    
conn.commit()        
cur.close()
conn.close()