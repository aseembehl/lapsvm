/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_classify.c                                       */
/*                                                                      */
/*   Classification Code for Latent SVM^struct                          */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 9.Nov.08                                                     */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include "svm_struct_latent_api.h"

void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *scorefile, STRUCT_LEARN_PARM *sparm);


int main(int argc, char* argv[]) {
  double *scores = NULL;
  long i;

  char testfile[1024];
  char modelfile[1024];
    char scoreFile[1024];
    FILE *fscore;

  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;
  LEARN_PARM lparm;
  KERNEL_PARM kparm;

  SAMPLE testsample;

  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,scoreFile,&sparm);
	fscore = fopen(scoreFile,"w");

  /* read model file */
  printf("Reading model..."); fflush(stdout);
  model = read_struct_model(modelfile, &sparm);
  printf("done.\n"); 

  /* read test examples */
	printf("Reading test examples..."); fflush(stdout);
  testsample = read_struct_examples(testfile,&sparm);
	printf("done.\n");

  init_struct_model(testsample,&model,&sparm,&lparm,&kparm);

  scores = classify_struct_example(testsample.examples[0].x,&model);
  for(i = 0; i < (testsample.examples[0].n_pos+testsample.examples[0].n_neg); i++){
    fprintf(fscore, "%0.5f\n", scores[i]);
  }
    
  fclose(fscore);

  free_struct_model(model,&sparm);

  return(0);

}


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *scorefile, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  sparm->custom_argc = 0;
  sparm->feature_size = 2405;

  for (i=1;(i<argc)&&((argv[i])[0]=='-');i++) {
    switch ((argv[i])[1]) {
      case '-': strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);i++; strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);break;  
      default: printf("\nUnrecognized option %s!\n\n",argv[i]); exit(0);    
    }
  }

  if (i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    exit(0);
  }

  strcpy(testfile, argv[i]);
	if(i+1<argc)
  	strcpy(modelfile, argv[i+1]);
	if(i+2<argc)
		strcpy(scorefile,argv[i+2]);

  parse_struct_parameters(sparm);

}
