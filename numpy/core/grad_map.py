
grad_map  = {}

def add_gradient_pair(non_grad_func, grad_func):
    grad_map[non_grad_func] = grad_func    

# def gradient_of(non_grad_func):
#     def decorator(grad_func):
#         def wrapper():
#             '''Add the function as key and its gradient function as value'''
#             grad_dict[non_grad_func] = grad_func
#             return grad_func
#         return wrapper
#     return decorator




