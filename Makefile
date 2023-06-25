IMAGE_NAME := barb
HOST_VOLUME := /home/bread/Documents/Barbershop
CONTAINER_VOLUME := /root/
COMMAND := /bin/bash -c "cd /root && python3 main.py --input_dir input/face --im_path1 16.png --im_path2 15.png --im_path3 117.png --sign realistic --smooth 5"

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: run
run:
	docker run -v $(HOST_VOLUME):$(CONTAINER_VOLUME) --gpus=all --rm -it $(IMAGE_NAME) $(COMMAND)

.PHONY: clean
clean:
	docker image rm $(IMAGE_NAME)
