for i in {1..6}
do
    bash /lfs/mlperf/common/intfix.bash
    echo "Run #$i"
    bash runtraining18.bash
    bash runtraining18.bash
    #bash runtraining18.bash
    sleep 42m  
    bash /lfs/mlperf/common/intfix.bash
    bash ../common/intfix.bash
    bash ../common/intfix.bash
done
