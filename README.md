## Design choices for Shapeflow:
1. Input of our_Operations class should only be lists, not any other objects to make it iterable
2. Input to the `forward` function of any operation should be the variable (changing) parameters of that operation. The constant ones are not.
3. The asserts involving the variable (changing) parameters (something like `x` or `input_tensor`) of the operation is repeated in its main body and its `forward` function. Other asserts are only in the main body of the operation (something like `axis` or `strides`). The main body ones, can catch errors before the session is called and the `forward` functions ones can do it after. 
4. The output of our_Operation is always a Tensor object
5. Placeholders, variables, constants, operations whenever declared are added to the default graph (object of class `our_Graph`). This is later used when session is called.
All operations on objects of class `our_Operation` (eg. __add__, __sub__, __getitem__) must return an object from same class, so that it can traced back to needed placeholders.
