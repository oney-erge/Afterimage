nproc
free -g | awk "NR==2{print \"RAM total=\"\$2\" avail=\"\$7}"
