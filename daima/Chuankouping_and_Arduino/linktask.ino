 //arduino的GND接串口屏或串口工具的GND,共地
 //arduino的9接串口屏或串口工具的RX
 //arduino的8接串口屏或串口工具的TX
 //arduino的5V接串口屏的5V,如果是串口工具,不用接5V也可以
 //根据自己的实际改为对应的串口,我这里接的是8和9
 #include <SoftwareSerial.h>
 SoftwareSerial TJC(8,9);
 #define FRAME_LENGTH 7

 int a;
 unsigned long nowtime;
 void setup() {
   // put your setup code here, to run once:
   //初始化串口
  Serial.begin(9600);
  TJC.begin(115200);

   //因为串口屏开机会发送88 ff ff ff,所以要清空串口缓冲区
   while (TJC.read() >= 0); //清空串口缓冲区
   TJC.print("page main\xff\xff\xff"); //发送命令让屏幕跳转到main页面
   nowtime = millis(); //获取当前已经运行的时间
 }


 void loop() {
   // put your main code here, to run repeatedly:
  char str[100];
  char incomingByte =0;
     if (Serial.available() > 0) {
    // 读取数据
    incomingByte = Serial.read();

 
   if (millis() >= nowtime + 8000) {
     nowtime = millis(); //获取当前已经运行的时间
    /*
     //用sprintf来格式化字符串，给n0的val属性赋值
     sprintf(str, "n0.val=%d\xff\xff\xff", a);
     //把字符串发送出去
     TJC.print(str);

     //用sprintf来格式化字符串，给t0的txt属性赋值
     sprintf(str, "t0.txt=\"现在是%d\"\xff\xff\xff", a);
     //把字符串发送出去
     TJC.print(str);
    */
  //page1
    //用sprintf来格式化字符串，触发b0的按下事件,直接把结束符整合在字符串中
        if (incomingByte == '0') {
      // 执行数据为 "0" 时的操作
      Serial.println("接收到: 0");
      // 可以添加其他控制逻辑，例如控制LED灯等
      sprintf(str, "click b3,1\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
     delay(50);  //延时50ms,才能看清楚点击效果
     //用sprintf来格式化字符串，触发b0的弹起事件,直接把结束符整合在字符串中
     sprintf(str, "click b3,0\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
    }

    else if (incomingByte == '1') {
      // 执行数据为 "1" 时的操作
      Serial.println("接收到: 1");
      // 可以添加其他控制逻辑
      sprintf(str, "click b1,1\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
     delay(50);  //延时50ms,才能看清楚点击效果
     //用sprintf来格式化字符串，触发b0的弹起事件,直接把结束符整合在字符串中
     sprintf(str, "click b1,0\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
    }

    else if (incomingByte == '2') {
      // 执行数据为 "2" 时的操作
      Serial.println("接收到: 2");
      // 可以添加其他控制逻辑
           //page 3
         //用sprintf来格式化字符串，触发b0的按下事件,直接把结束符整合在字符串中
     sprintf(str, "click b0,1\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
     delay(50);  //延时50ms,才能看清楚点击效果
     //用sprintf来格式化字符串，触发b0的弹起事件,直接把结束符整合在字符串中
     sprintf(str, "click b0,0\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
    }

    else if (incomingByte == '3') {
      // 执行数据为 "3" 时的操作
      Serial.println("接收到: 3");
      // 可以添加其他控制逻辑
           //page 4
         //用sprintf来格式化字符串，触发b0的按下事件,直接把结束符整合在字符串中
     sprintf(str, "click b2,1\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
     delay(50);  //延时50ms,才能看清楚点击效果
     //用sprintf来格式化字符串，触发b0的弹起事件,直接把结束符整合在字符串中
     sprintf(str, "click b2,0\xff\xff\xff");
     //把字符串发送出去
     TJC.print(str);
    }
        else {
      // 如果收到的是其他数据，执行此部分
      Serial.println("未知数据");
    }

     a++;
       }
     }
   //串口数据格式：
   //串口数据帧长度：7字节
   //帧头     参数1    参数2   参数3       帧尾
   //0x55     1字节   1字节    1字节     0xffffff
   //当参数是01时
   //帧头     参数1    参数2   参数3       帧尾
   //0x55     01     led编号  led状态    0xffffff
   //例子1：上位机代码  printh 55 01 01 00 ff ff ff  含义：1号led关闭
   //例子2：上位机代码  printh 55 01 04 01 ff ff ff  含义：4号led打开
   //例子3：上位机代码  printh 55 01 00 01 ff ff ff  含义：0号led打开
   //例子4：上位机代码  printh 55 01 04 00 ff ff ff  含义：4号led关闭

   //当参数是02或03时
   //帧头     参数1    参数2   参数3       帧尾
   //0x55     02/03   滑动值    00    0xffffff
   //例子1：上位机代码  printh 55 02 64 00 ff ff ff  含义：h0.val=100
   //例子2：上位机代码  printh 55 02 00 00 ff ff ff  含义：h0.val=0
   //例子3：上位机代码  printh 55 03 64 00 ff ff ff  含义：h1.val=100
   //例子4：上位机代码  printh 55 03 00 00 ff ff ff  含义：h1.val=0


   //当串口缓冲区大于等于一帧的长度时
   while (TJC.available() >= FRAME_LENGTH) {
     unsigned char ubuffer[FRAME_LENGTH];
     //从串口缓冲读取1个字节但不删除
     unsigned char frame_header = TJC.peek();
     //当获取的数据是包头(0x55)时
     if (frame_header == 0x55) {
       //从串口缓冲区读取7字节
       TJC.readBytes(ubuffer, FRAME_LENGTH);
       if (ubuffer[4] == 0xff && ubuffer[5] == 0xff && ubuffer[6] == 0xff) {
         if(ubuffer[1] == 0x01)
         {
           //下发的是LED信息
           sprintf(str, "msg.txt=\"led %d is %s\"\xff\xff\xff", ubuffer[2], ubuffer[3] ? "on" : "off");
           TJC.print(str);

         }else if(ubuffer[1] == 0x02)
         {
           //下发的是滑动条h0.val的信息
           sprintf(str, "msg.txt=\"h0.val is %d\"\xff\xff\xff", ubuffer[2]);
           TJC.print(str);

         }else if(ubuffer[1] == 0x03)
         {
           //下发的是滑动条h1.val的信息
           sprintf(str, "msg.txt=\"h1.val is %d\"\xff\xff\xff", ubuffer[2]);
           TJC.print(str);

         }

       }
     } else {
       TJC.read();  //从串口缓冲读取1个字节并删除
     }
   }
 }