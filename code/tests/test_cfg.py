"""Check guided target/JVP mathematics, null labels, and accumulation."""
import copy,unittest
import torch
from src.cfg_meanflow import make_loss_function,CFGMeanFlowUNet
from src.original_meanflow.recipe import original_meanflow_loss

class Field(torch.nn.Module):
    def __init__(self):
        super().__init__();self.a=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64));self.b=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64));self.d=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
    def forward(self,z,t,*,r,category,aug_cond=None):
        return self.a*z.square()+self.b*t[:,None,None,None]+self.d*(t-r)[:,None,None,None]*z+.13*(category+1)[:,None,None,None]

class CFGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123);self.x=torch.randn(4,3,4,4,dtype=torch.float64);self.e=torch.randn_like(self.x)
        self.r=torch.tensor([.1,.3,.5,.7],dtype=torch.float64);self.t=torch.tensor([.8,.3,.6,.9],dtype=torch.float64);self.c=torch.tensor([0,1,2,3]);self.mask=torch.tensor([False,True,False,True])
    def analytic(self,m,scale,mask):
        t=self.t[:,None,None,None];r=self.r[:,None,None,None];z=(1-t)*self.x+t*self.e;v=self.e-self.x
        null=(m.a*z.square()+m.b*t).detach()
        guided=torch.where(mask[:,None,None,None],v,scale*v+(1-scale)*null)
        # Analytic total derivative along (guided velocity, dr=0, dt=1).
        derivative=(2*m.a*z+m.d*(t-r))*guided+m.b+m.d*z
        target=(guided-(t-r)*derivative).detach()
        pred=m(z,self.t,r=self.r,category=torch.where(mask,-1,self.c))
        sq=(pred-target).float().square().flatten(1).sum(1)
        return (sq/(sq.detach()+1e-3).pow(.75)).mean()
    def test_guided_target_and_tangent_and_stopped_gradient(self):
        for scale in [1.,2.,3.]:
            a=Field();b=copy.deepcopy(a)
            actual,_=make_loss_function(guidance_scale=scale)(a,self.x,self.c,noise=self.e,r=self.r,t=self.t,drop_mask=self.mask);actual.backward()
            expected=self.analytic(b,scale,self.mask);expected.backward()
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-6)
    def test_scale_one_matches_original_mf(self):
        a=Field();b=copy.deepcopy(a);mask=torch.zeros(4,dtype=torch.bool)
        actual,_=make_loss_function(guidance_scale=1,label_dropout=0)(a,self.x,self.c,noise=self.e,r=self.r,t=self.t,drop_mask=mask);actual.backward()
        expected,_=original_meanflow_loss(b,self.x,self.c,noise=self.e,r=self.r,t=self.t);expected.backward()
        torch.testing.assert_close(actual,expected)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)
    def test_unconditional_examples_have_unguided_target(self):
        a=Field();mask=torch.ones(4,dtype=torch.bool)
        loss,stats=make_loss_function(guidance_scale=3)(a,self.x,self.c,noise=self.e,r=self.r,t=self.t,drop_mask=mask)
        torch.testing.assert_close(loss,self.analytic(a,1,mask));self.assertEqual(stats['guidance_delta_ms'].item(),0)
    def test_accumulation(self):
        a=Field();b=copy.deepcopy(a);fn=make_loss_function(guidance_scale=2)
        loss,_=fn(a,self.x,self.c,noise=self.e,r=self.r,t=self.t,drop_mask=self.mask);loss.backward()
        for i in [slice(0,2),slice(2,4)]:
            loss,_=fn(b,self.x[i],self.c[i],noise=self.e[i],r=self.r[i],t=self.t[i],drop_mask=self.mask[i]);(loss/2).backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-6)
    def test_guidance_target_is_eval_and_training_mode_restored(self):
        model=Field().train();modes=[]
        hook=model.register_forward_pre_hook(lambda m,args:modes.append(m.training))
        make_loss_function(guidance_scale=2)(model,self.x,self.c,noise=self.e,r=self.r,t=self.t,drop_mask=self.mask)
        hook.remove();self.assertFalse(modes[0]);self.assertTrue(all(modes[1:]));self.assertTrue(model.training)

    def test_null_is_zero_vector_and_class_zero_is_not_null(self):
        model=CFGMeanFlowUNet(model_channels=16,num_blocks=1,img_resolution=8).eval();seen=[]
        h=model.map_label.register_forward_pre_hook(lambda m,args:seen.append(args[0].clone()))
        with torch.no_grad():model(torch.randn(3,3,8,8),torch.ones(3),r=torch.zeros(3),category=torch.tensor([-1,0,150]))
        h.remove();self.assertEqual(seen[0][0].sum().item(),0);self.assertAlmostEqual(seen[0][1,0].item(),151**.5,places=5);self.assertAlmostEqual(seen[0][2,150].item(),151**.5,places=5)
        self.assertEqual(model.label_dropout,0)

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
