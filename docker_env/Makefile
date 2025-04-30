# make appropriate changes before running the scripts
# START OF CONFIG =====================================================================================================
IMAGE=img_name/dl-aio
CONTAINER=containter_name
AVAILABLE_GPUS='0,1,2,3'
LOCAL_JUPYTER_PORT=18888
LOCAL_TENSORBOARD_PORT=18006
PASSWORD=yourpassword
WORKSPACE=workspace_directory
# END OF CONFIG  ======================================================================================================

docker-resume:
	docker start -ai $(CONTAINER)

docker-run:
	docker run --restart=always --shm-size=8g --env=DISPLAY --gpus '"device=$(AVAILABLE_GPUS)"' -it -e PASSWORD=$(PASSWORD) -e JUPYTER_TOKEN=$(PASSWORD) -p $(LOCAL_JUPYTER_PORT):8888 -p \
		$(LOCAL_TENSORBOARD_PORT):6006 -v $(WORKSPACE):/notebooks --name $(CONTAINER) $(IMAGE)
		
docker-stop:
	docker stop $(CONTAINER)

docker-shell:
	docker exec -it $(CONTAINER) bash

docker-clean:
	docker rm $(CONTAINER)

docker-build:
	docker build -t $(IMAGE) -f Dockerfile .
	
docker-rebuild:
	docker build -t $(IMAGE) -f Dockerfile --no-cache --pull .

docker-push:
	docker push $(IMAGE)

docker-tensorboard:
	docker exec -it $(CONTAINER) tensorboard --logdir=logs
