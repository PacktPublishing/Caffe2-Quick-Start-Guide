#include "caffe2/core/init.h"

int main(int argc, char** argv)
{
    caffe2::GlobalInit(&argc, &argv);
    return 0;
}
