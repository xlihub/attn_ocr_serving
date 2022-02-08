OUTPUT="$(pip3 show labelme > config.txt)"
while read -r line
do
 if [[ $line == Location:* ]];
    then
    echo "$line"
    loc=${line#*Location: }
    echo "$loc"
 fi
done < config.txt
path='\labelme\widgets\'
filepath=${loc}${path}
echo "$filepath"
/bin/cp -r file_dialog_preview.py "$filepath"