../../third_party/llvm-build/Release+Asserts/bin/clang++ -g -funwind-tables --target=arm-linux-gnueabihf -march=armv7-a -mfloat-abi=hard -mtune=generic-armv7-a -mfpu=neon -mthumb -O2  -Wall -std=gnu++14 ./test-filter-alpha.cpp -o test-filter-alpha -nostdinc++ -isystem../../buildtools/third_party/libc++/trunk/include -isystem../../buildtools/third_party/libc++abi/trunk/include --sysroot=../../build/linux/raspbian_stretch_pi1-sysroot
