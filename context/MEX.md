# MATLAB MEX Interface for `f_hhh.c`

This file documents the original MATLAB gateway function (`mexFunction`) that was removed from `c_code/f_hhh.c` to allow for standalone compilation. This interface is required for calling the C function `hhh` directly from MATLAB.

## Gateway Function

```c
/***********************************************************************
**
**  Gateway function
**
***********************************************************************/

void mexFunction(
  int nlhs, mxArray *plhs[],
  int nrhs, const mxArray *prhs[])

{
/*vectors passed to the Matlab function*/
  double xbar,sd,rb1,bb2;

/*vector returned to the Matlab function*/
  double *vitype,*vgamma,*vdelta,*vxlam,*vxi,*vifault;
  double   itype,  gamma,  delta,  xlam,  xi,  ifault;


/*Right hand side of the MEX instruction*/
  xbar     = mxGetScalar(prhs[0]);   
  sd       = mxGetScalar(prhs[1]);
  rb1      = mxGetScalar(prhs[2]);
  bb2      = mxGetScalar(prhs[3]);

/*Left hand side of the MEX instruction */
  plhs[0]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);
  plhs[1]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);
  plhs[2]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);
  plhs[3]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);
  plhs[4]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);
  plhs[5]=mxCreateDoubleMatrix((int) 1,(int) 1,mxREAL);


  vgamma  = mxGetPr(plhs[0]); 
  vdelta  = mxGetPr(plhs[1]); 
  vxlam   = mxGetPr(plhs[2]); 
  vxi     = mxGetPr(plhs[3]); 
  vitype  = mxGetPr(plhs[4]);
  vifault = mxGetPr(plhs[5]);


/*Call the main computational routine*/
  hhh(xbar,sd,rb1,bb2,&itype,&gamma,&delta,&xlam,&xi,&ifault);

  *vgamma=gamma; *vdelta=delta; *vxlam=xlam; *vxi=xi; *vifault=ifault; *vitype=itype;

}
```
