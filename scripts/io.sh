cat /proc/diskstats | awk "\$3==\"sdd\" {print \"sectors_read=\" \$6}"
