Architecture
============

System Overview
---------------

.. uml::

   @startuml
   skinparam componentStyle rectangle

   package "SmallAIOS Unikernel" {
     [ONNX Runtime] as onnx
     [IPC (Zenoh-wire)] as ipc
     [Kernel Core] as kernel
     [Security / PQC] as security
     [Networking (TCP/IP)] as net

     package "Hardware Abstraction" {
       [x86-64 HAL] as x86
       [AArch64 HAL] as arm
       [NVIDIA GPU HAL] as gpu
     }
   }

   cloud "External" {
     [ONNX Model] as model
     [IPC Clients] as clients
   }

   model --> onnx : load
   clients --> ipc : pub/sub
   onnx --> kernel : memory, scheduling
   ipc --> kernel : syscalls
   ipc --> net : TCP transport
   kernel --> security : capabilities
   kernel --> x86 : x86-64
   kernel --> arm : AArch64
   onnx --> gpu : CUDA execution
   @enduml

Boot Sequence
-------------

.. uml::

   @startuml
   |Firmware/Bootloader|
   start
   :Load ELF kernel;
   :Jump to _start;

   |Assembly Entry|
   :Clear BSS;
   :Set up stack;
   :Call kernel_main;

   |Kernel Init|
   :Initialize serial/UART;
   :Print boot banner;
   :Initialize memory allocator;
   :Initialize scheduler;
   :Initialize security subsystem;
   :Initialize networking;
   :Initialize IPC;

   |Runtime|
   :Load ONNX model;
   :Start inference loop;
   :Serve IPC requests;
   stop
   @enduml
