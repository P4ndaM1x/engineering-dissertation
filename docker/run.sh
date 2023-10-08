docker build /mnt/d/studies/inżynierka/code/docker -t tensorflowongpu
docker run -it -v /mnt/d/studies/inżynierka/:/host -p 8888:8888 --gpus all tensorflowongpu
docker exec -it 4dc3a310de8b bash