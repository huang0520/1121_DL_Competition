echo "Please enter the image tag:"
read TAG

docker build \
-t $TAG \
--build-arg USERNAME=$USER \
--build-arg USER_UID=$(id -u) \
--build-arg USER_GID=$(id -g) \
.
