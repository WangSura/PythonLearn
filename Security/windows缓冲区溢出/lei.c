#include <stdio.h>
#include <windows.h>
#include <string.h>
#include <malloc.h>
int argc;
char FILENAME[] = "myoutfile";
void main()
{

    FILE *fd;
    long diff;
    char bufchar[100];

    char *buf1 = malloc(20); //高地址
    char *buf2 = malloc(20); //低地址

    long id1 = (long)buf1;
    long id2 = (long)buf2;

    diff = id2 - id1;
    strcpy(buf2, FILENAME);

    printf("信息显示---\n");
    printf("buf1存储地址:%p:\n", buf1);
    printf("buf2存储地址:%p ,存储内容为文件名:%s\n", buf2, buf2);
    printf("两地址间距离: %d个字节\n", diff);
    printf("显示信息结束---\n");

    if (argc < 2)
    {
        printf("请输入要写入文件%s的字符串\n", buf2);
        gets(bufchar);
        strcpy(buf1, bufchar);
    }
    else
    {
        printf("XXXXXXXXXXXX");
    }

    printf("----信息显示-----\n");
    printf("buf1存储内容: %s:\n", buf1);
    printf("buf2存储内容，%s:\n", buf2);
    printf("显示信息结束--\n");
    printf("将%s\n写入文件%s中\n", buf1, buf2);
    fd = fopen(buf2, "a");
    if (fd == NULL)
    {
        fprintf(stderr, "%s,打开错误\n", buf2);
    }
    fprintf(fd, "%s  \n", buf1);
    fclose(fd);

    getchar();
}