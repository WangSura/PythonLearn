#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
int k;
void fun(const char *input)
{
    char buf[8];
    strcpy(buf, input);
    k = (int)&input - (int)buf;
    printf("%s\n", buf);
}

void haha()
{
    printf("\nOK!success");
}

int main(int argc, char *argv[])
{
    printf("Address of foo=%p\n", fun);
    printf("Address of haha=%p\n", haha);
    void haha();
    int addr[4];
    char s[] = "FindK";
    fun(s);
    //printf("%d\n", k);
    int go = (int)&haha;
    //由于EIP地址是倒着表示的，所以首先把haha()函数的地址分离成字节
    addr[0] = (go << 24) >> 24;
    addr[1] = (go << 16) >> 24;
    addr[2] = (go << 8) >> 24;
    addr[3] = go >> 24;
    char ss[] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    for (int j = 0; j < 4; j++)
    {
        ss[k - j - 1] = addr[3 - j];
    }
    //fun(argv[1]);
    fun(ss);
    return 0;
}