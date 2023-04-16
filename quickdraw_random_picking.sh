#quickdraw 클래스 존재하는  폴더로감
folders=$(ls $train_folder)#! 존재하는 클래스 폴더이름 다 받아옴. 345개
for folder in $folders; do
  mkdir -p "/local_datasets/quickdraw_sub_ver1/val/$folder"
done
#이러면 val에 빈 폴더들 생성됌. 345개.
#우리가 사용ㅇ하려는 클래스 폴더를 val 폴더에 똑같이 만들어놓음 (아마 345개 다쓸꺼면 )

for folder in $folders; do
    find /local_datasets/quickdraw_sub_ver1/train/$folder -name "*.png" -type f -print0 | shuf -n 10000 -z | xargs -0 -I{} mv {} /local_datasets/quickdraw_sub_ver1/val/$folder
done


# Input the number of files to remove

num_files_to_remove=40000
# Specify the directory to remove PNG files from
folder_path="./"  # Replace with the actual path to your folder

# Validate the input
if ! [[ "$num_files_to_remove" =~ ^[0-9]+$ ]]; then
  echo "Error: Please enter a valid number."
  exit 1
fi

# Check if the specified directory exists
if ! [ -d "$folder_path" ]; then
  echo "Error: Directory \"$folder_path\" does not exist."
  exit 1
fi

# Find all PNG files in the directory
png_files=($(find "$folder_path" -type f -name "*.png"))

# Check if the number of PNG files is less than the number of files to remove
if [ "${#png_files[@]}" -le "$num_files_to_remove" ]; then
  echo "Number of PNG files in \"$folder_path\" is less than or equal to $num_files_to_remove. No files will be removed."
  exit 0
fi

# Shuffle the array of PNG files randomly
shuffled_files=($(shuf -e "${png_files[@]}"))

# Remove the specified number of files from the shuffled array
files_to_remove=("${shuffled_files[@]:0:num_files_to_remove}")

# Remove the selected files
for file in "${files_to_remove[@]}"; do
  rm "$file"
  echo "Removed file: $file"
done