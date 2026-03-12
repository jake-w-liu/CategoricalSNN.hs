#include <cstdlib>

#include <verilated.h>

#include "Vsnn12to7to5to3Flat.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn12to7to5to3Flat *top = new Vsnn12to7to5to3Flat;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

