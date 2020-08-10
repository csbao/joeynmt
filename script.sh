arr=()
arr+=('configs/viznet/2_2_0.1_128_128_128.yml')
arr+=('configs/viznet/2_2_0.1_128_128_64.yml')
arr+=('configs/viznet/2_2_0.1_128_64_256.yml')
arr+=('configs/viznet/2_2_0.1_64_128_128.yml')
arr+=('configs/viznet/2_2_0.1_64_128_64.yml')
arr+=('configs/viznet/2_2_0.1_64_64_256.yml')
arr+=('configs/viznet/2_2_0.2_128_128_128.yml')
arr+=('configs/viznet/2_2_0.1_128_128_256.yml')
arr+=('configs/viznet/2_2_0.1_128_64_128.yml')
arr+=('configs/viznet/2_2_0.1_128_64_64.yml')
arr+=('configs/viznet/2_2_0.1_64_128_256.yml')
arr+=('configs/viznet/2_2_0.1_64_64_128.yml')
arr+=('configs/viznet/2_2_0.1_64_64_64.yml')
i=$((0))
for filename in configs/viznet/attempt4/*.yml; do
  i=$(($i + 1));
  echo $i

  if [[ " ${arr[*]} " == *"$filename"* ]];
  then
      continue
  else
      python3 -m joeynmt train $filename;
  fi
    # ... rest of the loop body
done
