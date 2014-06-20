/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"

typedef struct sampleScore{
    int sample_idx;
    double sample_score;
} SAMPLE_SCORE;

// custom structure to store bounding box annotation for action classifiction problem.
typedef struct bbox {
	int min_x;
	int min_y;
	int width;
	int height;
} BBOX;


typedef struct sub_pattern {
  char file_name[1000];
  int *id_map;
  BBOX *boxes;
  int *annotations;
  SVECTOR **phis;
  int n_candidates;
  int label;
  
} SUB_PATTERN;

typedef struct pattern {
  /*
    Type definition for input pattern x
  */
    double example_cost; /* cost of individual example */

    SUB_PATTERN *x_is;
    long n_pos;
    long n_neg;
} PATTERN;

typedef struct label {
  /*
    Type definition for output label y
  */
  int *ranking;
  int *labels;
  
  long n_pos;
  long n_neg;
} LABEL;

typedef struct _sortStruct {
    double val;
    int index;
}  sortStruct;

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */
  int *h_is;
} LATENT_VAR;

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
  
  long n_subexamples; // no. of images in case of action classication
  long n_pos;
  long n_neg;
} EXAMPLE;


typedef struct sample {
  int n; // the number of examples. In case of AP-SVM there is only 1 example
  EXAMPLE *examples;
} SAMPLE;


typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;             /* number of examples */
} STRUCTMODEL;


typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  long newconstretrain;        /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* add your own variables */
  long feature_size; // dimension of feature vector
  int rng_seed; // random seed to intialise the values of latent variables
  
} STRUCT_LEARN_PARM;

