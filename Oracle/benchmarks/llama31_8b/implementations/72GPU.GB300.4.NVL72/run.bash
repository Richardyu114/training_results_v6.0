for i in {1..4}
do
#    bash /lfs/mlperf/common/intfix.bash
    echo "Run #$i"
    bash runtraining64.bash
    sleep 15m  
    bash ../common/intfix.bash
    bash ../common/intfix.bash
done
