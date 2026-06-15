for i in {1..5}
do
#    bash /lfs/mlperf/common/intfix.bash
    echo "Run #$i"
    bash runtraining18.bash
    bash runtraining18.bash
    bash runtraining18.bash
    bash runtraining18.bash
    sleep 50m  
    bash ../common/intfix.bash
    bash ../common/intfix.bash
done
