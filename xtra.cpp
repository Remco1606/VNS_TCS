/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#endif
 
#define nrn_init _nrn_init__xtra
#define _nrn_initial _nrn_initial__xtra
#define nrn_cur _nrn_cur__xtra
#define _nrn_current _nrn_current__xtra
#define nrn_jacob _nrn_jacob__xtra
#define nrn_state _nrn_state__xtra
#define _net_receive _net_receive__xtra 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define rx1 _p[0]
#define rx1_columnindex 0
#define rx2 _p[1]
#define rx2_columnindex 1
#define rx3 _p[2]
#define rx3_columnindex 2
#define rx4 _p[3]
#define rx4_columnindex 3
#define rx5 _p[4]
#define rx5_columnindex 4
#define rx6 _p[5]
#define rx6_columnindex 5
#define x _p[6]
#define x_columnindex 6
#define y _p[7]
#define y_columnindex 7
#define z _p[8]
#define z_columnindex 8
#define type _p[9]
#define type_columnindex 9
#define order _p[10]
#define order_columnindex 10
#define ex	*_ppvar[0].get<double*>()
#define _p_ex _ppvar[0].literal_value<void*>()
#define area	*_ppvar[1].get<double*>()
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 static int hoc_nrnpointerindex =  0;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {"setdata_xtra", _hoc_setdata},
 {0, 0}
};
 /* declare global and static user variables */
#define is6 is6_xtra
 double is6 = 0;
#define is5 is5_xtra
 double is5 = 0;
#define is4 is4_xtra
 double is4 = 0;
#define is3 is3_xtra
 double is3 = 0;
#define is2 is2_xtra
 double is2 = 0;
#define is1 is1_xtra
 double is1 = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"is1_xtra", "miliamps"},
 {"is2_xtra", "miliamps"},
 {"is3_xtra", "miliamps"},
 {"is4_xtra", "miliamps"},
 {"is5_xtra", "miliamps"},
 {"is6_xtra", "miliamps"},
 {"rx1_xtra", "ohm"},
 {"rx2_xtra", "ohm"},
 {"rx3_xtra", "ohm"},
 {"rx4_xtra", "ohm"},
 {"rx5_xtra", "ohm"},
 {"rx6_xtra", "ohm"},
 {"x_xtra", "1"},
 {"y_xtra", "1"},
 {"z_xtra", "1"},
 {"type_xtra", "1"},
 {"order_xtra", "1"},
 {"ex_xtra", "millivolts"},
 {0, 0}
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"is1_xtra", &is1_xtra},
 {"is2_xtra", &is2_xtra},
 {"is3_xtra", &is3_xtra},
 {"is4_xtra", &is4_xtra},
 {"is5_xtra", &is5_xtra},
 {"is6_xtra", &is6_xtra},
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 static void _ba1(Node*_nd, double* _pp, Datum* _ppd, Datum* _thread, NrnThread* _nt) ;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, Memb_list*, int);
static void nrn_state(NrnThread*, Memb_list*, int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"xtra",
 "rx1_xtra",
 "rx2_xtra",
 "rx3_xtra",
 "rx4_xtra",
 "rx5_xtra",
 "rx6_xtra",
 "x_xtra",
 "y_xtra",
 "z_xtra",
 "type_xtra",
 "order_xtra",
 0,
 0,
 0,
 "ex_xtra",
 0};
 extern Node* nrn_alloc_node_;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	rx1 = 0;
 	rx2 = 0;
 	rx3 = 0;
 	rx4 = 0;
 	rx5 = 0;
 	rx6 = 0;
 	x = 0;
 	y = 0;
 	z = 0;
 	type = 0;
 	order = 0;
 	_prop->param = _p;
 	_prop->param_size = 11;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 	_ppvar[1] = &nrn_alloc_node_->_area; /* diam */
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _xtra_reg() {
	int _vectorized = 0;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nullptr, nullptr, nullptr, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 11, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "pointer");
  hoc_register_dparam_semantics(_mechtype, 1, "area");
 	hoc_reg_ba(_mechtype, _ba1, 11);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 xtra xtra.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 /* BEFORE BREAKPOINT */
 static void _ba1(Node*_nd, double* _pp, Datum* _ppd, Datum* _thread, NrnThread* _nt)  {
    _p = _pp; _ppvar = _ppd;
  v = NODEV(_nd);
 ex = is1 * rx1 + is2 * rx2 + is3 * rx3 + is4 * rx4 + is5 * rx5 + is6 * rx6 ;
   }

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   ex = is1 * rx1 + is2 * rx2 + is3 * rx3 + is4 * rx4 + is5 * rx5 + is6 * rx6 ;
   }

}
}

static void nrn_init(NrnThread* _nt, Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_state(NrnThread* _nt, Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "xtra.mod";
    const char* nmodl_file_text = 
  ": $Id: xtra.mod,v 1.4 2014/08/18 23:15:25 ted Exp ted $\n"
  ": 2018/05/20 Modified by Aman Aberra \n"
  "\n"
  "NEURON {\n"
  "	SUFFIX xtra\n"
  "	RANGE rx1, rx2, rx3, rx4, rx5, rx6: (es = max amplitude of the potential)		\n"
  "	RANGE x, y, z, type, order\n"
  "	GLOBAL is1, is2, is3, is4, is5, is6: (stim = active electrode, amp = stimulation amplitude)\n"
  "	POINTER ex \n"
  "}\n"
  "\n"
  "PARAMETER {	\n"
  "	rx1 = 0 (ohm) : mV/mA\n"
  "	rx2 = 0 (ohm) : mV/mA\n"
  "	rx3 = 0 (ohm) : mV/mA\n"
  "	rx4 = 0 (ohm) : mV/mA\n"
  "	rx5 = 0 (ohm) : mV/mA\n"
  "	rx6 = 0 (ohm) : mV/mA\n"
  "	x = 0 (1) : spatial coords\n"
  "	y = 0 (1)\n"
  "	z = 0 (1)		\n"
  "	type = 0 (1) : numbering system for morphological category of section - unassigned is 0\n"
  "	order = 0 (1) : order of branch/collateral. \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (millivolts)\n"
  "	ex (millivolts)\n"
  "	is1 (miliamps)\n"
  "	is2 (miliamps)\n"
  "	is3 (miliamps)\n"
  "	is4 (miliamps)\n"
  "	is5 (miliamps)\n"
  "	is6 (miliamps)\n"
  "	area (micron2)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	ex = is1*rx1+is2*rx2+is3*rx3+is4*rx4+is5*rx5+is6*rx6\n"
  "}\n"
  "\n"
  "\n"
  "BEFORE BREAKPOINT { : before each cy' = f(y,t) setup\n"
  "  	ex = is1*rx1+is2*rx2+is3*rx3+is4*rx4+is5*rx5+is6*rx6\n"
  "}\n"
  "\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
