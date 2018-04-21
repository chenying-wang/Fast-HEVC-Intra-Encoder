#include <iostream>

// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/public/session.h"

using namespace std;
// using namespace tensorflow;
// using namespace tensorflow::ops;

#ifndef __SESSIONHANDLER__
#define __SESSIONHANDLER__

class SessionHandler
{
private:
  // ClientSession* session;

public:
  SessionHandler();
  virtual ~SessionHandler();
};

#endif // __SESSIONHANDLER__
