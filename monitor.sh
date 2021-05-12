# Record process start time to a log file
# Note: Replace -p argument with process ID for your training script
ps -o lstart= -p 11379 > watch_pretrain_11379.log

# check process every 2 seconds using watch command
# while process is running, record current time to log file
watch "ps -o lstart= -p 11379 && \
	(date | tee -a watch_segplusclass_11379.log)"
