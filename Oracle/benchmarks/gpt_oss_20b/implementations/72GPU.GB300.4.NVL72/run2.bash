for i in {1..11}
do
#    bash /lfs/mlperf/common/intfix.bash
    echo "Run #$i"
    bash runtraining2.bash
    #sleep 15m  
    #bash ../common/intfix.bash
    #bash ../common/intfix.bash
done
