#bin/#This script sets up the model, sets the tf model looking for input
f=$1
INFILE='in_list'
READINFILE='out_file'
OUTFILE='answers.txt'
echo -n "$f" >> $OUTFILE
echo "$f.wav"
echo "$f.wav" > $INFILE
sleep 2
RESULT=`cat $READINFILE`
echo ",$RESULT" >> $OUTFILE
echo "Done file $index"
sleep 0.1



