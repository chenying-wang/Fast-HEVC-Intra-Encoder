#include <iostream>

#include "TLibCommon/CommonDef.h"

// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/public/session.h"

// using namespace tensorflow;
// using namespace tensorflow::ops;

#ifndef __SESSIONWRAPPER__
#define __SESSIONWRAPPER__

class SessionWrapper
{
private:
  // ClientSession* session;
  Bool *m_pbIsSplit;

public:
  SessionWrapper();
  virtual ~SessionWrapper();
  Void init(UInt uiNumOfCus);
  Bool *infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth);
};

#endif // __SESSIONWRAPPER__
