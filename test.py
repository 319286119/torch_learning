#print('niub')

'''s1=input('请输入密码：')
if s1=='1':
    s2='正确'
else:
    s2="错误"
print(s2)'''

'''s=d=f=1
print(s,f)'''

'''x=y=5
x=x+3
x,y,=y,x
print(x)'''

#print(max(3,4,7,9,3.5))

'''a=2
a//=6
print(a)'''

#print(round(3.1415926,3))

#print(abs(1+2j))

#print(type(9.0))

#print('''niu
b''')'''
#print('''niu
b''')'''
#print("sftwteserdtg"[0:4:3])

#s="学校\t\t\t学生人数\t地址\t\n华北努比高校\t1000\t11"
'''s="111\t222"
print(s)'''
#print("1"not in"123")
#s='good good study'
#print(s.capitalize())n b
#as='good good study'
#print(s.endswith('stu',5,-3))

'''print('{0:>30,}'.format(123456))'''
                #字符串格式化

'''name='张三'
#score=input('qingshuru:')
print(name+"成绩为"+'{0:-^30}'.format(input('qingsuru:')))'''
'''print("学校为{1},{0:_^10}".format(111,"nnn"))'''

'''x=eval(input("qingshuru"))
print(x,type(x))'''
'''niub=12
print(eval("niub"))'''
'''print(eval("1+2"))'''
#eval（）的用法1.去除“”，改变数据类型  2.执行命令   3.加减法执行

''''print(12,34,sep="#")'''        #sep()表示以括号内的的标识填充间隙

#math库的引入采用import形式

'''import math
print(type(math.fabs(-1)))'''

#列表
'''x=[1,2,3,4]
print(x[0])'''

'''a,b,c=input("请输入数据：").split(",")
print(a,b,c)'''     #split（“”）可以为三个不同的变量分别赋值，括号中的为字符串


'''x=1
print(id(x))'''

#列表的加减法
a=[1,2,3]
b=[4,5]
'''print(a[0:3:2])'''
'''print(1 in a)'''#1是否在a中
'''print(a.index(3,0,0))'''#index()看3是否在a的0到0中
'''print(a.insert(1,2))'''#将2插入列表使其变为第1个
'''print(a.count(1))'''#a中有几个1


#自动生成符合一定规律的代码，使用range()函数，如从1到100的等差数列
'''print(list(range(1,100)))'''

                                    #元组一旦创建不可修改 ，其可视为不变的列表，但其表示方式与列表不同
'''s=(1,2,3)
print(type(s))'''
'''s=(1,)
print(s)'''#元组只有一个值时都好不能省略
'''s=tuple("python")
print(s)'''
#字典
'''s={"学校":"js"}
print(type(s))'''
'''s=dict(学校="jianshan")
       #键   值
print(s["学校"])#返回键对应的值'''
#集合
'''s={1,2,3}
print(type(s))'''


#数据类型之间的转化
 #字符串转为列表
'''x=list("nnnn")
print(x)'''
 #列表转为字符串
'''x=("1","2","3")
x1="".join(x)
print(x1)'''
#字典转化为列表
'''y={"1001":"wang001"}
x=list(y.keys())
print(x)
x=list(y.values())+list(y.keys())
print(x)'''


#列表转字典
'''y=[1,2,3]
x=['w','s','d']
z=zip(x,y)
z=dict(z)
print(z)'''

#random库的使用
    #import random
'''x=random.random()
print(x)'''#随机生成【0，1.0）之间的数
'''x=random.uniform(1,10)
print(x)'''#随机生成设定范围之间的小数
'''x=random.randint(1,10)
print(x)'''#随机生成设定范围之间的整数

'''import random
x={input("请输入学生a的姓名:"):input("请输入a的成绩："),"李四":80,"王五":input("请输入王五成绩：")}
y=["张三","李四","王五"]
z=random.choice(y)
print("学生为：",z,x.get(z))'''


#选择结构
'''a=int(input("请输入数字一："))
b=int(input("请输入数字二："))
print("输入的数字分别为{0},{1}".format(a,b))
if a>b:
    print("根据升序排序后为：{0},{1}".format(b,a))
else:
    print("根据升序排序后为：{0}，{1}".format(a,b))'''


'''import random
x=random.randint(0,100)
y=input("请输入1到100中的某个数：")
if y==x:
    print("right")
else:
    print("false")
print(x)'''

#calendar。time.datetime库的运用
import calendar
'''print(calendar.prcal(2020))'''#用prcal()函数打印2020年的日历


'''sum=0
i=1
while i<=100:
    sum+=i
    i+=1
print(sum)'''


'''x=1
a=eval(input())
b=eval(input())
print(a,type(b),type(a))'''


'''i=1
sum=0
while i<=100:
    sum+=i
    i+=1
print(sum)'''

'''i=1
x=1
while i<=5:
    print(x*i)
    i+=1'''

'''x="hello world"
for i in range(len(x)):
    print(x[i])'''

'''for num in[1,2,3,]:
    print(num)'''

'''i=0
for x in range(0,101,1):
    i+=x
print(i)'''


#求n的阶乘
'''n=int(input())
x=1
for i in range(1,n+1,1):
    x*=i
print(x)'''

'''n=int(input())
x=1
s=0
for i in range(1,n+1):
   x*=i
   s+=x
print(s)'''#算一到n的阶乘的和

#嵌套循环
'''一元硬币换为多少,两分，一分,五分,'''
'''i=0#五分
j=0#两分
k=0#一分
count=0
for i in range(0,21):
    for j in range(0,51):
        k=100-5*i-2*j
        if k>0:
            count+=1
print(count)'''#count为总数

'''i=0#五角
j=0#两角
k=0#一角
count=0
for i in range(1,3):
    for j in range(1,6):
        for k in range(1,11):
         k=10-5*i-2*j
         if k>0:
            count+=1
print(count)'''
                #count德总数)


#用while的方式写
'''i=0#五分
j=0#两分
k=0#一分
count=0
while i<=20:
    j=0
    while j<=50:
        k=100-5*i-2*j
        if k>0:
            count+=1
        j+=1
    i+=1
print(count)'''


#break和continue
'''s=0
i=0
while i<10:#可替换为while true
    s+=i
    if s>10:
        break
    i+=1
print("i={0:d},sum={1:d}".format(i,s))
print(s)'''

#输出从二到一百内的所有素数，每行五个
'''count = 0#方便每行五个所用的变量
for num in range(2, 101):
    is_prime = True#is_prime是判断素数用的
    for i in range(2, num):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:#从此开始实现每行五个
        print(num,end=" ")
        count += 1
        if count % 5 == 0:
            print("")'''


'''import random
x=random.randint(0,101)
while x==x:
    y = int(input('请输入：'))
    if y>x:
        print("dale")
    elif y<x:
        print('xiaole')
    elif y==x:
        print('duile')
        break'''

'''f=open('','w')
print(f)'''

'''import os
os.startfile('notepad.exe')'''


'''def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]

weights = [2, 3, 4, 7]
values = [1, 3, 5, 9]
capacity = 10
print(knapsack(weights, values, capacity))'''





'''weights = [2, 3, 4, 7]
values = [1, 3, 5, 9]
capacity = 10

n = len(weights)
max_value = 0
num=0
m=[]
for i in range(0,n+1):
    if num<=10:
      i+=1
      m.append(i)
      num=weights[i]+num
      print(m)
    if num>10:
        break'''



'''def kyx(nums,i):
    if i==len(nums)-1:
        return 1
    max_len=1
    for j in range(i+1,len(nums)):
        if nums[j]>nums[i]:
            max_len=max(max_len,kyx(nums,j)+1)
    return max_len'''


'''weight = [2, 3, 4, 7]
value = [1, 3, 5, 9]
n = len(weight)
max_value2=0
max_value1=0
for i in range(0,n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            total_weight2 = weight[i] + weight[j] + weight[k]
            total_value2 = value[i] + value[j] + value[k]
            if total_value2>=max_value2:
                max_value2=total_value2
            if total_weight2 <= 10:
                print(total_value2)


for o in range(0,n):
    for p in range(o+1,n):
        total_weight1 = weight[o] + weight[p]
        total_value1 = value[o] + value[p]
        if total_weight1 <= 10 and total_value1 >= max_value1:
            max_value1 = total_value1
print(max_value1)'''


'''n=3#行数
i=0
j=0
k=0
while(i<3):
    while(j<=2-i):
        print(" ")
        j+=1
    while(k<2*i-1):
        print("*")
i+=1


n = 3
i = 0
while (i < n):
    j = 0
    while (j < n - i - 1):
        print(" ")
        j += 1
    k = 0
    while (k < 2 * i + 1):
        print("*")
        k += 1
    print()
    i += 1
i = n - 2
while (i >= 0):
    j = 0
    while (j < n - i - 1):
        print(" ")
        j += 1
    k = 0
    while (k < 2 * i + 1):
        print("*")
        k += 1
    print()
    i -= 1'''
#x,y,z=map(int,input("输入").split())

'''import copy

# 原始列表，包含一个子列表
original_list = [1, 2, [3, 4]]

# 浅复制
shallow_copy = copy.copy(original_list)

# 修改原始列表中的子列表
original_list[2][0] = 99

print("原始列表:", original_list)
print("浅复制列表:", shallow_copy)'''

'''import copy

# 原始列表，包含一个子列表
original_list = [1, 2, [3, 4]]

# 深复制
deep_copy = copy.deepcopy(original_list)

# 修改原始列表中的子列表
original_list[2][0] = 99

print("原始列表:", original_list)
print("深复制列表:", deep_copy)'''

'''tup=(1,2,3)
print(tup[1])'''

#import calendar
#print(calendar.prmonth(2025,4))

''''from math import sqrt
num=int(input("请输入"))
j=2
while j<=int(sqrt(num)):
    if (num%j)==0:
        print("not")
        break
else:
    print("is")'''

'''try:
    num1,num2=map(int,input("输入:").split())
    print(num1/num2)
except ZeroDivisionError:
    print("ZeroDivisionError")
except ValueError:
    print("ZeroDivisionError")
else:
    print("smart")'''


'''def add(*args):
    result=0
    for i in args:
        result+=i
    return result

if __name__=="__main__":
    print(add(1,2))'''

'''users={}
def sign_up():
    user_name=input("输入用户名")
    while user_name in users.keys():
        user_name=input("已有用户，重新输入")
    password=input("输入密码")
    users[user_name]=password
    print("注册成功")

def sign_in():
    user_name=input("输入用户名")
    while user_name not in users:
        user_name=input("用户不存在，重新输入")
    password=input("请输入密码")
    count=0
    while password!=users[user_name]:
        password=input("密码错误，请重新输入")
        count+=1
    else:
        print("密码正确，登录成功")

if __name__=="__main__":
    while True:
        cmd=input("请输入:1、sign up 2、sign in")
        if cmd!="1"and cmd!="2":
            cmd=input("重新输入:")
        if cmd=='1':
            sign_up()
        if cmd=='2':
            sign_in()'''

'''def add(x):
    x+=1

x=1
add(x)
print(x)'''
#不可变类型做参数不可在函数内被改变实参，可变参数则可以改变实参

'''def add(*args):#接受位置参数，放入一个元组
    ret=0
    for num in args:
        ret+=num
    return ret

def dir_add(**args):#接受关键字参数，放入一个字典
    key_ret=''
    val_ret=0
    for num in args.keys():
        key_ret+=num
    for num in args.values():
        val_ret+=num
    return {val_ret,key_ret}

print(add(1,2,3,4))
print("result is {0}".format(dir_add(a=1,b=2)))
print(dir_add(a=1,b=2))'''

'''x=5
def add():
    global x
    x=3
    print(x)
    def inner():
        print(x)
    inner()

add()'''

'''numbers=[1,3,4,7]
ret=map(lambda x:x**2,numbers)
new_numbers=list(ret)
print(new_numbers)'''

'''import matplotlib.pyplot as plt
import numpy as np
x=range(10,20)
y=x
plt.plot(x,y,'r')
plt.bar(x,y,color='b')
plt.show()
x=range(2,26,2)
y=[20,21,22,22,24,22,23,23,23,22,22,21]
plt.figure(figsize=(10,8),dpi=80)

plt.plot(x,y)
plt.xticks(np.arange(2,25,1.5))
#或者是plt.xticks(i/2 for i in range(2,25,1))
plt.show()'''

'''import matplotlib.pyplot as plt
import random

x=range(2,25,2)
y=[random.randint(15,30) for i in range(0,12)]
plt.plot(x,y)
_xlabels=["{}:00".format(i) for i in range(2,25,2)]
plt.xticks(x,_xlabels)
plt.show()'''#画一个柱形统计图

'''import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,2*np.pi,1000)

y=np.sin(x)
plt.plot(x,y)
plt.title("y=sin(x)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()'''#画二维图像

'''import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)
z = np.sin(x) + np.cos(x)

# 创建一个 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲线
ax.plot(x, y, z)

ax.set_title("3D Curve of y = sin(x) and z = sin(x) + cos(x)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()'''#画三维图像

'''import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,2*np.pi,1000)
y=np.sin(x)
z=np.sin(x)+np.cos(y)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(x,y,z)

plt.show()'''

'''class Person:

    def __init__(self,name,age):
        self.__name=name
        self.__age=age

    def info(self):
        print("姓名=",self.__name)
        print("年龄=",self.__age)

me=Person("ky",19)
me.info()'''#封装，_为protected,__为private

'''class Animal:
    def __init__(self,name):
        self.__name=name

    def speak(self):
        print("hh")

class Dog(Animal):
    def __init__(self,name,age):
        super().__init__(name)
        self.__age=age

    def bark(self):
        print("www")

dog=Dog("旺财",2)
dog.bark()
dog.speak()'''#继承

'''class Person:
    __num=0
    def __init__(self,name):
        self.__name=name
        Person.__num+=1

    @classmethod
    def speak(cls):
        print(cls.__num)'''#装饰器

'''class Base:
    def __init__(self):
        pass

class Parent1(Base):
    def __init__(self):
        super().__init__()
        self.__name="1"

class Parent2(Base):
    def __init__(self):
        super().__init__()
        self.name="2"

class Son(Parent1,Parent2):
    def __init__(self):
        super().__init__()#根据mro，先调用Parent1
        Parent2.__init__()#指定调用'''






