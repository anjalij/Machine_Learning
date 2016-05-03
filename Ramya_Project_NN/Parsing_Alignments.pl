#!/usr/bin/perl

use Data::Dumper qw(Dumper);



my $str = " >T0L8K8
\par ------------------------------------------------------------
\par ------------------------------------------------------------
\par \hich\af6\dbch\af31505\loch\f6 ------------------------------------------------CLQILKNEKYYL
\par \hich\af6\dbch\af31505\loch\f6 KCGYVGVINKSQTDINNKISFKEIELKEKNWFLNS-VYKNLD-NIGNKYLVNKVKELFVS
\par \hich\af6\dbch\af31505\loch\f6 RCRD\hich\af6\dbch\af31505\loch\f6 LFPCLKSVICEKICVLEDELKSY---------------------------------
\par ------------------------------------------------------------
\par ------------------------------------------------------------
\par ------------------------------------------------------------
\par ------------------------------------------------------------
\par ----------------
\par \hich\af6\dbch\af31505\loch\f6 >E9KHJ0";

my @words = split '>', $str;
my $noSeq= $#words+1;

for($in=1;l $in<=$noSeq;$in++){
	my$s=@words[$in];
		$s=~ s/[^0-9]//g;
		print $s
}

