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

    char *buf1 = malloc(20); //�ߵ�ַ
    char *buf2 = malloc(20); //�͵�ַ

    long id1 = (long)buf1;
    long id2 = (long)buf2;

    diff = id2 - id1;
    strcpy(buf2, FILENAME);

    printf("��Ϣ��ʾ---\n");
    printf("buf1�洢��ַ:%p:\n", buf1);
    printf("buf2�洢��ַ:%p ,�洢����Ϊ�ļ���:%s\n", buf2, buf2);
    printf("����ַ�����: %d���ֽ�\n", diff);
    printf("��ʾ��Ϣ����---\n");

    if (argc < 2)
    {
        printf("������Ҫд���ļ�%s���ַ���\n", buf2);
        gets(bufchar);
        strcpy(buf1, bufchar);
    }
    else
    {
        printf("XXXXXXXXXXXX");
    }

    printf("----��Ϣ��ʾ-----\n");
    printf("buf1�洢����: %s:\n", buf1);
    printf("buf2�洢���ݣ�%s:\n", buf2);
    printf("��ʾ��Ϣ����--\n");
    printf("��%s\nд���ļ�%s��\n", buf1, buf2);
    fd = fopen(buf2, "a");
    if (fd == NULL)
    {
        fprintf(stderr, "%s,�򿪴���\n", buf2);
    }
    fprintf(fd, "%s  \n", buf1);
    fclose(fd);

    getchar();
}