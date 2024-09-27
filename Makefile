KOS_CFLAGS+= -g -std=c99 -Og -I$(KOS_BASE)/utils
TARGET = cloth.elf
OBJS = cloth.o 

all: rm-elf $(TARGET) cloth.bin 1st_read.bin

include $(KOS_BASE)/Makefile.rules

rm-elf:
	-rm -f $(TARGET) romdisk.*

$(TARGET): $(OBJS) romdisk.o
	kos-c++ -o $(TARGET) $(OBJS)romdisk.o -lpng -ljpeg -lkmg -lz -lkosutils -lm


DTTEXTURES:=$(shell find assets/texture -name '*.png'| sed -e 's,assets/\(.*\)/\([a-z_A-Z0-9]*\).png,romdisk/\1/\2.dt,g')
# DTTEXTURES+=$(shell find assets/texture -name '*.jpg'| sed -e 's,assets/\(.*\)/\([a-z_A-Z0-9]*\).jpg,romdisk/\1/\2.dt,g')
$(info $(DTTEXTURES))

# TEXDIR_PAL4=romdisk/texture/pal4
# $(TEXDIR_PAL4):
# 	mkdir -p $@

# $(TEXDIR_PAL4)/%.dt: assets/texture/pal4/%.png $(TEXDIR_PAL4)
# 	pvrtex -f PAL4BPP -c -i $< -o $@

TEXDIR_PAL8=romdisk/texture/pal8
$(TEXDIR_PAL8):
	mkdir -p $@

TEXDIR_RGB565_VQ_TW=romdisk/texture/rgb565_vq_tw
$(TEXDIR_RGB565_VQ_TW):
	mkdir -p $@

$(TEXDIR_RGB565_VQ_TW)/%.dt: assets/texture/rgb565_vq_tw/%.png $(TEXDIR_RGB565_VQ_TW)
	${KOS_BASE}/utils/pvrtex/pvrtex -f RGB565 -c -i $< -o $@

cloth.bin: cloth.elf
	sh-elf-objcopy -R .stack -O binary cloth.elf cloth.bin

1st_read.bin: cloth.bin
	$(KOS_BASE)/utils/scramble/scramble cloth.bin ./iso/1st_read.bin

romdisk.img: $(DTTEXTURES)
	$(KOS_GENROMFS) -f romdisk.img -d romdisk -v

romdisk.o: romdisk.img
	$(KOS_BASE)/utils/bin2o/bin2o romdisk.img romdisk romdisk.o

run: $(TARGET)
	$(KOS_LOADER) $(TARGET)

dist:
	rm -f $(OBJS) romdisk.o romdisk.img
	$(KOS_STRIP) $(TARGET)

clean:
	-rm -f $(TARGET) $(OBJS) romdisk.*

