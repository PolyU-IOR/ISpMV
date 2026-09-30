#pragma once
#include <cmath>
#include <random>
#include <vector>
// Shared frozen numerical vector protocol; no route or timing decisions.
static void vector_input(std::vector<double> &x,int trial) {
    std::mt19937_64 generator(20260910+trial);
    std::uniform_real_distribution<double> uniform(-1,1);
    std::normal_distribution<double> normal(0,1);
    for(size_t c=0;c<x.size();++c) {
        switch(trial) {
        case 0:x[c]=.5+((c*17+3)%101)/101.;break;
        case 1:x[c]=uniform(generator);break;
        case 2:x[c]=normal(generator);break;
        case 3:x[c]=c%2?-1:1;break;
        case 4:x[c]=1;break;
        case 5:x[c]=c==x.size()/3?1:0;break;
        case 6:x[c]=uniform(generator)*std::pow(10.,int(c%25)-12);break;
        case 7:x[c]=0;break;
        default:x[c]=std::sin(double(c)*.713)*1e-12;
        }
    }
}
