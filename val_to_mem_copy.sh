#!/bin/sh

# PLEASE, do not put a slash at the end of the directories
# it will NOT work

PERCENTAGE=100
SOURCE_DIR="/mnt/data/imagenet/original/val" # <- no slash at end
TARGET_DIR="/dev/shm/imagenet/val"

mapfile -t files < <( find $SOURCE_DIR -type f)
number_to_copy=$(( $PERCENTAGE * ${#files[@]} / 100))
echo "Copying $number_to_copy files of ${#files[@]} total"

files=()

for dir in $(find $SOURCE_DIR -type d | sed -n "s|^${SOURCE_DIR}/||p"); do 
    mkdir -p ${TARGET_DIR}/$dir
done

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

find $SOURCE_DIR -type f | sed -n "s|^${SOURCE_DIR}/||p" \
                         | sort \
                         | shuf --random-source=<(get_seeded_random ${SEED}) -n $number_to_copy \
                         | while read f; do
                             cp ${SOURCE_DIR}/$f ${TARGET_DIR}/$f
                         done
