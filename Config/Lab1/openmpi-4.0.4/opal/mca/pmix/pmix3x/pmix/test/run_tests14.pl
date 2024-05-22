#!/usr/bin/env perl
#
# Copyright (c) 2019      Intel, Inc.
#
# Copyright (c) 2019      Cisco Systems, Inc.  All rights reserved
# $COPYRIGHT$
#
# Additional copyrights may follow

use strict;

my @tests = ("-n 4 --ns-dist 3:1 --fence \"[db | 0:0-2;1:0]\"",
             "-n 4 --ns-dist 3:1 --fence \"[db | 0:;1:0]\"",
             "-n 4 --ns-dist 3:1 --fence \"[db | 0:;1:]\"",
             "-n 4 --ns-dist 3:1 --fence \"[0:]\"",
             "-n 4 --ns-dist 3:1 --fence \"[b | 0:]\"",
             "-n 4 --ns-dist 3:1 --fence \"[d | 0:]\" --noise \"[0:0,1]\"",
             "-n 4 --job-fence -c",
             "-n 4 --job-fence",
             "-n 2 --test-publish",
             "-n 2 --test-spawn",
             "-n 2 --test-connect",
             "-n 5 --test-resolve-peers --ns-dist \"1:2:2\"",
             "-n 5 --test-replace 100:0,1,10,50,99",
             "-n 5 --test-internal 10",
             "-s 1 -n 2 --job-fence",
             "-s 1 -n 2 --job-fence -c");

my $test;
my $cmd;
my $output;
my $status = 0;
my $testnum;
my $timeout_cmd = "";

# We are running tests against the build tree (vs. the installation
# tree).  Autogen gives us a full list of all possible component
# directories in PMIX_COMPONENT_LIBRARY_PATHS.  Iterate through each
# of those directories: 1) to see if there is actually a component
# built in there, and 2) to turn it into an absolute path name.  Then
# put the new list in the "mca_bast_component_path" MCA parameter env
# variable so that the MCA base knows where to find all the
# components.
my @myfullpaths;
my $mybuilddir = "/home/forever/openmpi-4.0.4/opal/mca/pmix/pmix3x/pmix";
my $mypathstr = "src/mca/bfrops/v12:src/mca/bfrops/v20:src/mca/bfrops/v21:src/mca/bfrops/v3:src/mca/common/dstore:src/mca/gds/ds12:src/mca/gds/ds21:src/mca/gds/hash:src/mca/pdl/pdlopen:src/mca/pdl/plibltdl:src/mca/pif/bsdx_ipv4:src/mca/pif/bsdx_ipv6:src/mca/pif/linux_ipv6:src/mca/pif/posix_ipv4:src/mca/pif/solaris_ipv6:src/mca/pinstalldirs/config:src/mca/pinstalldirs/env:src/mca/plog/default:src/mca/plog/stdfd:src/mca/plog/syslog:src/mca/pnet/opa:src/mca/pnet/tcp:src/mca/pnet/test:src/mca/preg/native:src/mca/psec/dummy_handshake:src/mca/psec/munge:src/mca/psec/native:src/mca/psec/none:src/mca/psensor/file:src/mca/psensor/heartbeat:src/mca/pshmem/mmap:src/mca/ptl/tcp:src/mca/ptl/usock";
my @splitstr = split(':', $mypathstr);
foreach my $path (@splitstr) {
    # Note that the component is actually built in the ".libs"
    # subdirectory.  If the component wasn't built, that subdirectory
    # will not exist, so don't save it.
    my $fullpath = $mybuilddir . "/" . $path . "/.libs";
    push(@myfullpaths, $fullpath)
        if (-d $fullpath);
}
my $mymcapaths = join(":", @myfullpaths);
$ENV{'PMIX_MCA_mca_base_component_path'} = $mymcapaths;

my $wdir = $mybuilddir . "/test";
chdir $wdir;

$testnum = $0;
$testnum =~ s/.pl//;
$testnum = substr($testnum, -2);
$test = @tests[$testnum];

# find the timeout or gtimeout cmd so we can timeout the
# test if it hangs
my @paths = split(/:/, $ENV{PATH});
foreach my $p (@paths) {
    my $fullpath = $p . "/" . "gtimeout";
    if ((-e $fullpath) && (-f $fullpath)) {
        $timeout_cmd = $fullpath . " --preserve-status -k 35 30 ";
        last;
    } else {
        my $fullpath = $p . "/" . "timeout";
        if ((-e $fullpath) && (-f $fullpath)) {
            $timeout_cmd = $fullpath . " --preserve-status -k 35 30 ";
            last;
        }
    }
}

$cmd = $timeout_cmd . " ./pmix_test " . $test . " 2>&1";
print $cmd . "\n";
$output = `$cmd`;
print $output . "\n";
print "CODE $?\n";
$status = "$?";

exit($status >> 8);