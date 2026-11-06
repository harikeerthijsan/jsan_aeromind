import pytest, time
from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        r=client.get('/health')
        assert r.status_code==200 and r.json()['ai_inference']=='not_configured'

def test_registration_clusters_need_multiple_images():
    import numpy as np
    from app.registration import _clusters
    rows=[{'position':np.array([0.,0.,0.]),'weight':.9,'confidence':.8,'media_id':1,'detection_id':10,'point_count':5},
          {'position':np.array([.03,0.,0.]),'weight':.8,'confidence':.82,'media_id':2,'detection_id':11,'point_count':7},
          {'position':np.array([2.,0.,0.]),'weight':.95,'confidence':.9,'media_id':3,'detection_id':12,'point_count':8}]
    result=_clusters(rows,.1)
    assert len(result)==1 and result[0]['media_count']==2 and set(result[0]['detection_ids'])=={10,11}

def test_registration_suppresses_duplicate_2d_boxes_only_within_class():
    from types import SimpleNamespace as NS
    from app.registration import _nms
    box=lambda i,name,confidence,x1,x2:NS(id=i,class_name=name,confidence=confidence,x1=x1,y1=10,x2=x2,y2=90)
    boxes=[box(1,'rru',.92,10,90),box(2,'rru',.80,12,88),box(3,'rru',.85,110,170),box(4,'sector_antenna',.75,12,88)]
    kept=_nms(boxes)
    assert {b.id for b in kept}=={1,3,4}  # duplicate RRU removed; neighbour and other class retained

def test_registration_merges_multiview_fragments_not_neighbours(monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from app import registration as reg
    det=lambda i,x1,x2:NS(id=i,confidence=.9,x1=x1,y1=10,x2=x2,y2=110)
    def candidate(position,views):
        return {'class_name':'sector_antenna','position':np.array(position,float),'confidence':.9,'media_count':2,
                'evidence_count':2,'point_count':100,'spread':.02,'detection_ids':[d.id for _,d in views],
                'size_w':.3,'size_h':1.2,'size_d':.15,'yaw':0.,'box_method':'dense','box_points':80,
                'fit_iou':.8,'_views':views}
    first=candidate([1,3,0],[('a',det(1,10,50)),('b',det(2,12,52))])
    fragment=candidate([1.02,3,.01],[('a',det(3,11,51)),('b',det(4,13,53))])
    neighbour=candidate([1,3,.4],[('a',det(5,70,110)),('b',det(6,72,112))])
    monkeypatch.setattr(reg,'fit_to_detections',lambda c,*args,**kwargs:c)
    out=reg.merge_duplicate_candidates([first,fragment,neighbour],{}, {},np.eye(4))
    assert len(out)==2
    assert sorted(c['media_count'] for c in out)==[2,2]
    assert sorted(len(c['detection_ids']) for c in out)==[2,2]

def test_registration_uses_shape_aware_ladder_validation():
    from app.registration import candidate_is_valid
    base={'class_name':'ladder','box_method':'dense','media_count':4,'confidence':.423,'fit_iou':.303,
          'size_w':.077,'size_h':.667,'size_d':.029}
    assert candidate_is_valid(base)
    assert not candidate_is_valid(base|{'size_h':.12})  # compact false positive, not a ladder

def _registration_fixture():
    import uuid
    from app.database import SessionLocal
    from app.models import Inspection, Model3D, RegistrationCandidate, Site
    with SessionLocal() as db:
        site=Site(code='REG-'+uuid.uuid4().hex[:10],name='Registration review test');db.add(site);db.flush()
        inspection=Inspection(site_id=site.id,label='3D review');db.add(inspection);db.flush()
        model=Model3D(inspection_id=inspection.id,status='completed',quality={});db.add(model);db.flush()
        rows=[RegistrationCandidate(model_id=model.id,inspection_id=inspection.id,site_id=site.id,class_name='rru',x=i*.2,y=0,z=1,
               confidence=.9,status='needs_review',evidence_count=5,media_count=5,point_count=100,spread=.01,detection_ids=[10+i],
               size_w=.05,size_h=.1,size_d=.03,yaw=0,box_method='dense',box_points=80,fit_iou=.8) for i in range(2)]
        db.add_all(rows);db.commit();return model.id,inspection.id,[row.id for row in rows]

def test_registration_review_api_supports_single_and_bulk_decisions():
    model_id,_,ids=_registration_fixture()
    with TestClient(app) as client:
        r=client.patch(f'/api/v1/registrations/{ids[0]}',json={'status':'approved'})
        assert r.status_code==200 and r.json()['status']=='approved'
        r=client.post(f'/api/v1/models/{model_id}/registrations/review',json={'candidate_ids':ids,'status':'needs_adjustment'})
        assert r.status_code==200 and r.json()['updated_count']==2
        assert {row['status'] for row in r.json()['candidates']}=={'needs_adjustment'}
        r=client.patch(f'/api/v1/registrations/{ids[0]}/geometry',json={'x':.3,'y':-.2,'z':1.4,'yaw':7.0})
        assert r.status_code==200 and (r.json()['x'],r.json()['y'],r.json()['z'])==(.3,-.2,1.4)
        assert -3.1416<=r.json()['yaw']<=3.1416
        assert client.patch(f'/api/v1/registrations/{ids[0]}',json={'status':'approved'}).status_code==200
        assert client.patch(f'/api/v1/registrations/{ids[0]}/geometry',json={'x':0,'y':0,'z':0}).status_code==409
        assert client.patch(f'/api/v1/registrations/{ids[0]}',json={'status':'invalid'}).status_code==422

def test_registration_rebuild_preserves_approved_id_and_updates_unreviewed_in_place():
    import numpy as np
    from sqlalchemy import select
    from app.database import SessionLocal
    from app.main import reconcile_registrations
    from app.models import Inspection, Model3D, RegistrationCandidate
    model_id,inspection_id,ids=_registration_fixture()
    with SessionLocal() as db:
        model,inspection=db.get(Model3D,model_id),db.get(Inspection,inspection_id)
        approved,mutable=db.get(RegistrationCandidate,ids[0]),db.get(RegistrationCandidate,ids[1]);approved.status='approved';db.commit()
        def generated(position,detection_id):return {'class_name':'rru','position':np.array(position,float),'confidence':.92,'spread':.01,
            'media_count':6,'evidence_count':6,'point_count':120,'detection_ids':[detection_id],'size_w':.05,'size_h':.1,'size_d':.03,
            'yaw':0.,'box_method':'dense','box_points':90,'fit_iou':.82}
        stats=reconcile_registrations(db,model,inspection,[generated([.01,1.01,0],10),generated([.22,1.02,0],11),generated([.5,1,0],12)]);db.commit()
        rows=list(db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model_id)))
        assert stats=={'created':1,'updated':1,'preserved':1,'removed':0}
        assert len(rows)==3 and db.get(RegistrationCandidate,approved.id).status=='approved'
        assert db.get(RegistrationCandidate,mutable.id).x==pytest.approx(.22) and db.get(RegistrationCandidate,mutable.id).status=='needs_review'

def test_registration_rebuild_preserves_needs_adjustment_geometry():
    import numpy as np
    from app.database import SessionLocal
    from app.main import reconcile_registrations
    from app.models import Inspection, Model3D, RegistrationCandidate
    model_id,inspection_id,ids=_registration_fixture()
    with SessionLocal() as db:
        model,inspection=db.get(Model3D,model_id),db.get(Inspection,inspection_id);row=db.get(RegistrationCandidate,ids[0])
        row.status='needs_adjustment';row.x=.35;db.commit()
        generated={'class_name':'rru','position':np.array([.02,1,0],float),'confidence':.92,'spread':.01,'media_count':6,
                   'evidence_count':6,'point_count':120,'detection_ids':[10],'size_w':.05,'size_h':.1,'size_d':.03,
                   'yaw':0,'box_method':'dense','box_points':80,'fit_iou':.8}
        result=reconcile_registrations(db,model,inspection,[generated]);db.commit()
        kept=db.get(RegistrationCandidate,ids[0])
        assert result['preserved']==1 and kept.status=='needs_adjustment' and kept.x==pytest.approx(.35)

def test_sites_and_dashboard():
    with TestClient(app) as client:
        assert client.get('/api/v1/sites').status_code==200
        assert 'assets' in client.get('/api/v1/dashboard').json()

def test_process_without_detector():
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Test run'}).json()
        r=client.post(f"/api/v1/inspections/{i['id']}/process")
        assert r.status_code==202 and r.json()['status']=='awaiting_worker'
        assert client.get(f"/api/v1/inspections/{i['id']}/detections").json()==[]

def test_approve_image_annotations():
    from app.database import SessionLocal
    from app.models import Detection, Job
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'QC run'}).json()
        m=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('t.png',b'\x89PNG\r\n\x1a\n','image/png')}).json()
        with SessionLocal() as db:  # stand-in for a detector run
            job=Job(inspection_id=i['id'],kind='detection',status='completed',stage='detection');db.add(job);db.flush()
            dets=[Detection(job_id=job.id,inspection_id=i['id'],media_id=m['id'],class_id=k,class_name=n,confidence=.9,x1=10,y1=10,x2=50,y2=50,image_width=100,image_height=80,model_name='test') for k,n in [(0,'qc_antenna'),(1,'qc_rru'),(1,'qc_rru')]]
            db.add_all(dets);db.commit();keep,edit,drop,job_id=dets[0].id,dets[1].id,dets[2].id,job.id
        assert client.get(f"/api/v1/media/{m['id']}/annotations").json()=={'review':None,'annotations':[]}
        boxes=[{'detection_id':keep,'class_name':'qc_antenna','x1':10,'y1':10,'x2':50,'y2':50},
               {'detection_id':edit,'class_name':'qc_antenna','x1':12,'y1':10,'x2':50,'y2':50},
               {'class_name':'qc_new_part','x1':95,'y1':70,'x2':60,'y2':200}]  # reversed + out of bounds: normalised and clamped
        r=client.put(f"/api/v1/media/{m['id']}/annotations",json={'workflow':'ai_detection','job_id':job_id,'image_width':100,'image_height':80,'boxes':boxes})
        assert r.status_code==200,r.text
        body=r.json();a=body['annotations']
        assert body['review']['status']=='approved' and body['review']['box_count']==3
        assert [x['source'] for x in a]==['model','model_edited','manual']
        assert (a[2]['x1'],a[2]['y1'],a[2]['x2'],a[2]['y2'])==(60,70,95,80)
        assert a[2]['class_id'] not in (a[0]['class_id'],) and a[2]['confidence'] is None
        status={d['id']:d['review_status'] for d in client.get(f"/api/v1/inspections/{i['id']}/detections").json()}
        assert status=={keep:'accepted',edit:'corrected',drop:'rejected'}
        assert 'qc_new_part' in [c['name'] for c in client.get('/api/v1/classes').json()]
        # re-approving replaces the set; QC can approve an image as having no objects
        r=client.put(f"/api/v1/media/{m['id']}/annotations",json={'workflow':'manual_qc','image_width':100,'image_height':80,'boxes':[]})
        assert r.json()['annotations']==[] and r.json()['review']['workflow']=='manual_qc'
        assert len(client.get(f"/api/v1/inspections/{i['id']}/reviews").json())==1
        bad=client.put(f"/api/v1/media/{m['id']}/annotations",json={'image_width':100,'image_height':80,'boxes':[{'class_name':'x','x1':5,'y1':5,'x2':5.5,'y2':40}]})
        assert bad.status_code==422 and len(client.get(f"/api/v1/media/{m['id']}/annotations").json()['annotations'])==0

def _drone_jpeg(orientation=6):
    """64x48 JPEG carrying the metadata a DJI camera writes: EXIF+GPS+MakerNote, XMP, ICC, IPTC, MPF index, comment."""
    import io
    from PIL import Image, ImageCms
    ex=Image.Exif();ex[0x010F]='DJI';ex[0x0110]='FC6310';ex[0x0112]=orientation
    ex[0x8769]={0x927C:b'DJI makernote \x01\x02'};ex[0x8825]={1:'N',2:(17.0,23.0,1.5),3:'E',4:(78.0,28.0,2.25)}
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    b=io.BytesIO();Image.new('RGB',(64,48),(128,128,128)).save(b,'JPEG',exif=ex,icc_profile=icc,quality=90);j=b.getvalue()
    seg=lambda m,body:bytes([0xFF,m])+(len(body)+2).to_bytes(2,'big')+body
    xmp=b'http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><rdf:Description drone-dji:GimbalYawDegree="-87.5"/></x:xmpmeta>'
    return j[:2]+seg(0xE1,xmp)+seg(0xED,b'Photoshop 3.0\0IPTC')+seg(0xE2,b'MPF\0stale offsets')+seg(0xFE,b'survey 42')+j[2:]

def test_render_keeps_jpeg_metadata(tmp_path):
    pytest.importorskip('PIL')
    import io
    from PIL import Image, ImageOps
    from app.annotate import render, jpeg_segments
    src=_drone_jpeg();p=tmp_path/'drone.jpg';p.write_bytes(src)
    # ultralytics reads JPEGs upright, so boxes come in the 48x64 display frame of this orientation-6 image
    out,mime=render(p,[{'class_id':1,'class_name':'rru','confidence':.91,'x1':5,'y1':5,'x2':30,'y2':40}],seen_size=(48,64))
    assert mime=='image/jpeg'
    meta=[s for m,s in jpeg_segments(src)[0] if m in (0xE0,0xE1,0xED,0xFE) or (m==0xE2 and s[4:].startswith(b'ICC_PROFILE'))]
    assert all(s in out for s in meta) and b'MPF\0stale' not in out  # copied byte-for-byte; stale MPF index dropped
    im=Image.open(io.BytesIO(out));ex=im.getexif()
    assert im.size==(64,48) and ex[0x0112]==6 and ex[0x010F]=='DJI' and ex.get_ifd(0x8825)[1]=='N' and ex.get_ifd(0x8769)[0x927C]==b'DJI makernote \x01\x02'
    up=ImageOps.exif_transpose(im).convert('RGB')
    r,g,b=up.getpixel((5,20));assert g>r+20  # green rru outline where the box is in the upright frame
    assert up.getpixel((40,60))==up.getpixel((44,62))  # background untouched

def test_render_keeps_png_and_tiff_metadata(tmp_path):
    pytest.importorskip('PIL')
    import io
    from PIL import Image, PngImagePlugin
    from app.annotate import render
    box=[{'class_id':0,'class_name':'antenna','x1':2,'y1':2,'x2':20,'y2':20}]
    info=PngImagePlugin.PngInfo();info.add_text('Author','pilot');info.add_itxt('XML:com.adobe.xmp','<x:xmpmeta>drone-dji:FlightYawDegree="12" °</x:xmpmeta>')
    pe=Image.Exif();pe[0x010F]='DJI';Image.new('RGBA',(40,30),(10,20,30,255)).save(tmp_path/'a.png',pnginfo=info,exif=pe,dpi=(300,300))
    q=Image.open(io.BytesIO(render(tmp_path/'a.png',box)[0]))
    assert q.mode=='RGBA' and q.text==Image.open(tmp_path/'a.png').text and q.getexif()[0x010F]=='DJI' and round(q.info['dpi'][0])==300
    te=Image.Exif();te[271]='DJI';te[700]=b'<x:xmpmeta/>';te.get_ifd(0x8825).update({1:'N',2:(17.0,23.0,1.5)});te[0x8825]=0
    Image.new('RGB',(40,30),(10,20,30)).save(tmp_path/'a.tif',exif=te)
    r=Image.open(io.BytesIO(render(tmp_path/'a.tif',box)[0]))
    assert r.tag_v2[271]=='DJI' and r.tag_v2[700]==b'<x:xmpmeta/>' and r.getexif().get_ifd(0x8825)[1]=='N'

def test_annotated_download_endpoints():
    pytest.importorskip('PIL')
    import io, zipfile
    from PIL import Image
    from app.database import SessionLocal
    from app.models import Detection, Job
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Annotated output'}).json()
        m=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('DJI_0001.JPG',_drone_jpeg(),'image/jpeg')}).json()
        with SessionLocal() as db:
            job=Job(inspection_id=i['id'],kind='detection',status='completed',stage='detection');db.add(job);db.flush()
            db.add(Detection(job_id=job.id,inspection_id=i['id'],media_id=m['id'],class_id=0,class_name='sector_antenna',confidence=.8,x1=4,y1=4,x2=40,y2=50,image_width=48,image_height=64,model_name='test'));db.commit()
        r=client.get(f"/api/v1/media/{m['id']}/annotated")
        assert r.status_code==200 and r.headers['x-annotation-source']=='model' and r.headers['x-box-count']=='1'
        assert 'DJI_0001_detections.jpg' in r.headers['content-disposition']
        assert Image.open(io.BytesIO(r.content)).getexif().get_ifd(0x8825)[1]=='N' and b'GimbalYawDegree' in r.content
        assert client.get(f"/api/v1/media/{m['id']}/annotated?source=approved").status_code==404
        client.put(f"/api/v1/media/{m['id']}/annotations",json={'workflow':'manual_qc','image_width':48,'image_height':64,'boxes':[]})
        r=client.get(f"/api/v1/media/{m['id']}/annotated")
        assert r.headers['x-annotation-source']=='approved' and r.headers['x-box-count']=='0'
        z=client.get(f"/api/v1/inspections/{i['id']}/annotated.zip?source=model")
        assert z.status_code==200
        with zipfile.ZipFile(io.BytesIO(z.content)) as f:
            assert f.namelist()==['DJI_0001_detections.jpg'] and b'drone-dji' in f.read('DJI_0001_detections.jpg')

def test_vendor_tags_suggestions_and_csv(monkeypatch):
    pytest.importorskip('PIL')
    from app import vendor
    with TestClient(app) as client:
        info=client.get('/api/v1/vendors').json()
        assert info['classes']==['sector_antenna','microwave_dish'] and 'Ericsson' in info['manufacturers']['sector_antenna']
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Vendor tags'}).json()
        m=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('DJI_0002.JPG',_drone_jpeg(1),'image/jpeg')}).json()
        boxes=[{'class_name':'sector_antenna','x1':2,'y1':2,'x2':30,'y2':40,'manufacturer':'Ericsson','product_model':'AIR 6449','vendor_source':'ai','vendor_confidence':.9,'vendor_evidence':'logo'},
               {'class_name':'microwave_dish','x1':32,'y1':2,'x2':60,'y2':30,'manufacturer':'Huawei','vendor_source':'reviewer','vendor_confidence':.5},
               {'class_name':'bolts','x1':40,'y1':35,'x2':50,'y2':45,'manufacturer':'Ericsson'}]
        a=client.put(f"/api/v1/media/{m['id']}/annotations",json={'workflow':'manual_qc','image_width':64,'image_height':48,'boxes':boxes}).json()['annotations']
        assert [(x['manufacturer'],x['product_model'],x['vendor_source'],x['vendor_confidence']) for x in a]==[('Ericsson','AIR 6449','ai',.9),('Huawei',None,'reviewer',None),(None,None,None,None)]
        csv_text=client.get(f"/api/v1/inspections/{i['id']}/annotations.csv").text
        assert 'DJI_0002.JPG,' in csv_text and 'Ericsson,AIR 6449,ai,0.9' in csv_text
        seen={}
        monkeypatch.setattr(vendor,'status',lambda:'ready')
        monkeypatch.setattr(vendor,'suggest',lambda im,cls,refs=():seen.update(size=im.size,cls=cls,refs=[r[0] for r in refs]) or {'manufacturer':'Unknown','confidence':.1})
        r=client.post(f"/api/v1/media/{m['id']}/vendor-suggestion",json={'class_name':'microwave_dish','x1':32,'y1':2,'x2':60,'y2':30,'image_width':64,'image_height':48})
        assert r.status_code==200 and r.json()['manufacturer']=='Unknown' and seen['cls']=='microwave_dish' and seen['size']==(36,34)  # 28px box + 15% padding, clipped at the image edge
        assert 'Huawei' not in seen['refs']  # the reviewer-tagged box itself is not offered as its own reference
        client.post(f"/api/v1/media/{m['id']}/vendor-suggestion",json={'class_name':'microwave_dish','x1':2,'y1':30,'x2':30,'y2':47,'image_width':64,'image_height':48})
        assert 'Huawei' in seen['refs']  # ...but it is a reference for other dishes
        assert client.get('/api/v1/vendors').json()['reviewer_references']['microwave_dish']>=1
        assert client.post(f"/api/v1/media/{m['id']}/vendor-suggestion",json={'class_name':'bolts','x1':1,'y1':1,'x2':9,'y2':9,'image_width':64,'image_height':48}).status_code==422
        monkeypatch.setattr(vendor,'status',lambda:'unreachable')
        assert client.post(f"/api/v1/media/{m['id']}/vendor-suggestion",json={'class_name':'sector_antenna','x1':1,'y1':1,'x2':9,'y2':9,'image_width':64,'image_height':48}).status_code==503

def test_structure_decision_logic(monkeypatch):
    pytest.importorskip('PIL')
    from PIL import Image
    from app import vendor
    im=Image.new('RGB',(40,80),'white');refs=[('NEC',im),('NEC',im),('Ceragon',im)]
    monkeypatch.setattr(vendor,'compare_model',lambda:'thinking-model')  # don't depend on a local Ollama
    def run(answers):
        it=iter(answers);monkeypatch.setattr(vendor,'_chat',lambda *a,**k:(next(it),1.0));return vendor.suggest(im,'microwave_dish',refs)
    logo_unknown={'manufacturer':'Unknown','confidence':.1,'evidence':'plain'}
    describe=lambda guess='Unknown',gc=0:{'features':{'dish_type':'shrouded'},'own_knowledge_guess':guess,'own_knowledge_confidence':gc,'reasoning':'r'}
    compare=lambda ok,sim:{'same_equipment_type':True,'matching_features':['outline'],'differing_features':[] if ok else ['ODU position'],'same_product':'yes' if ok else 'no','similarity':sim,'reasoning':'v'}
    r=run([{'manufacturer':'Huawei','confidence':.95,'evidence':'logo'}])
    assert (r['manufacturer'],r['basis'])==('Huawei','logo_text')  # a legible logo wins; no structure pass
    r=run([logo_unknown,describe(),compare(True,.9),compare(False,.2)])  # one call per manufacturer: NEC, then Ceragon
    assert (r['manufacturer'],r['basis'],r['confidence'])==('NEC','reference_match',.6) and r['reference_vendors']==['NEC','Ceragon']  # capped below logo evidence
    r=run([logo_unknown,describe(),compare(False,.3),compare(True,.8)])
    assert r['manufacturer']=='Ceragon'
    r=run([logo_unknown,describe(),compare(False,.3),compare(False,.2)])
    assert (r['manufacturer'],r['basis'])==('Unknown','none') and 'No reference matched' in r['evidence']
    r=run([logo_unknown,describe(),compare(True,.9),compare(True,.85)])
    assert r['manufacturer']=='Unknown' and 'Ambiguous' in r['evidence'] and r['ambiguous']  # two close matches: don't pick one...
    assert [(c['manufacturer'],c['similarity']) for c in r['candidates']]==[('NEC',.9),('Ceragon',.85)]  # ...offer both to the reviewer
    def flaky(answers):  # a comparison that never answers is skipped, not fatal
        it=iter(answers)
        def chat(*a,**k):
            x=next(it)
            if x is None: raise RuntimeError('VLM returned no usable JSON')
            return x,1.0
        monkeypatch.setattr(vendor,'_chat',chat);return vendor.suggest(im,'microwave_dish',refs)
    r=flaky([logo_unknown,describe(),None,compare(True,.8)])
    assert r['manufacturer']=='Ceragon' and 'NEC gave no answer' in r['evidence']
    monkeypatch.setattr(vendor,'compare_model',lambda:None)  # thinking model missing: skip, don't trust the instruct model
    r=run([logo_unknown,describe()])
    assert r['manufacturer']=='Unknown' and 'Structure comparison skipped' in r['evidence'] and not r['candidates']
    monkeypatch.setattr(vendor,'compare_model',lambda:'thinking-model')
    refs[:]=[('ANDREW',im),('CommScope',im),('Kathrein',im)]  # brands of one company are compared together: two calls, not three
    r=run([logo_unknown,describe(),compare(True,.9),compare(False,.3)])
    assert (r['manufacturer'],r['basis'])==('CommScope','reference_match') and r['reference_vendors']==['CommScope','Ericsson'] and 'ANDREW brand' in r['evidence']
    refs[:]=[('NEC',im),('NEC',im),('Ceragon',im)]
    r=run([logo_unknown,describe('Ericsson',.8),compare(False,.3),compare(False,.2)])
    assert r['manufacturer']=='Unknown' and r['structure_hint']=='Ericsson' and 'Unverified hint' in r['evidence']  # model knowledge never becomes the tag

def test_reference_library_import(tmp_path, monkeypatch):
    pytest.importorskip('PIL')
    import io, csv
    from PIL import Image
    from app import vendor, refs_import
    monkeypatch.setattr(vendor,'VENDOR_REFERENCE_DIR',tmp_path)
    def jpg(w,h):
        b=io.BytesIO();Image.new('RGB',(w,h),'white').save(b,'JPEG');return b.getvalue()
    got=[]
    def fake_fetch(url,timeout=30):
        got.append(url)
        if 'missing' in url: raise OSError('404')
        return jpg(600,900)
    monkeypatch.setattr(refs_import,'fetch',fake_fetch)
    man=tmp_path/'manifest.csv'
    man.write_text('category,manufacturer,model,view,url,page,status,downloaded,notes\n'
                   'Sector_Antennas,ANDREW,NNHH-65B-R4-V1,front,https://x/a.jpg/thumbnail,https://p,ok,NO,\n'
                   'Sector_Antennas,ANDREW,NNHH-65B-R4-V1,bottom_straight,https://x/b.jpg,https://p,ok,NO,\n'
                   'Sector_Antennas,ANDREW,NNHH-65B-R4-V1,side_installed,https://x/c.jpg,https://p,ok,NO,\n'
                   'Microwave_Dishes,Aviat_Networks,AND-VHLP6,rear_three_quarter,https://x/missing.jpg,https://p,ok,NO,check model\n',encoding='utf-8')
    rows=refs_import.import_manifest(man,log=lambda *a:None)
    assert got[0]=='https://x/a.jpg/medium'  # thumbnail links are upgraded
    assert [r['status'][:6] for r in rows]==['ok','ok','ok','failed'] and rows[3]['manufacturer']=='Aviat'
    lib=vendor.library('sector_antenna')
    assert {r['view']:r['match'] for r in lib}=={'front':True,'bottom_straight':False,'side_installed':True}  # connector ends: reviewer only
    assert [p.name for _,p in vendor.folder_references('sector_antenna')]==['NNHH-65B-R4-V1__side_installed.jpg','NNHH-65B-R4-V1__front.jpg']  # drone-like views first
    assert 'ANDREW' in vendor.manufacturers('sector_antenna')
    (tmp_path/'sector_antenna'/'Huawei').mkdir();Image.new('RGB',(100,200)).save(tmp_path/'sector_antenna'/'Huawei'/'aau.jpg')  # added by hand
    assert ('Huawei','aau.jpg') in [(m,p.name) for m,p in vendor.folder_references('sector_antenna')]
    with TestClient(app) as client:
        r=client.get('/api/v1/vendor-references',params={'class_name':'sector_antenna','manufacturer':'andrew'}).json()
        assert len(r)==3 and all(x['url'].startswith('/vendor-refs/sector_antenna/ANDREW/') for x in r) and sum(x['match'] for x in r)==2
    assert list(csv.DictReader(open(tmp_path/'references.csv',encoding='utf-8')))[3]['status'].startswith('failed')

def test_condition_rules_and_second_opinion(monkeypatch):
    pytest.importorskip('PIL')
    from PIL import Image
    from app import condition, vendor
    n=condition.normalise
    d=lambda t,s,e:{'type':t,'severity':s,'extent_percent':e,'location':'bracket','evidence':'brown flakes'}
    r=n({'image_quality':'good','grade':'Good','health_score':95,'confidence':.8,'defects':[d('rust','severe',60)],'summary':'s'})
    assert r['grade']=='Critical' and r['health_score']<=30 and r['rust_percent']==60  # a severe defect can't be Good
    r=n({'image_quality':'good','grade':'Good','health_score':95,'confidence':.8,'defects':[d('rust','moderate',20)],'summary':'s'})
    assert (r['grade'],r['rust_percent'])==('Poor',20)
    assert n({'image_quality':'good','grade':'Poor','health_score':40,'confidence':.8,'defects':[],'summary':'s'})['grade']=='Fair'  # no defect, no Poor
    r=n({'image_quality':'insufficient','grade':'Good','health_score':90,'confidence':.9,'defects':[],'summary':'s'})
    assert (r['grade'],r['health_score'])==('Unknown',None)
    calls=[];im=Image.new('RGB',(200,300),'white')
    monkeypatch.setattr(vendor,'_chat',lambda *a,**k:calls.append(1))
    r=condition.assess_verified(im,'bolts',(11,15))
    assert r['grade']=='Unknown' and r['skipped'] and not calls  # too small: the model isn't asked
    monkeypatch.setattr(vendor,'compare_model',lambda:'thinking-model')
    def chat_seq(answers):
        it=iter(answers);monkeypatch.setattr(vendor,'_chat',lambda *a,**k:(next(it),1.0))
    rusty={'image_quality':'good','grade':'Fair','health_score':80,'confidence':.7,'defects':[d('rust','minor',10)],'summary':'Light rust.'}
    clean={'image_quality':'good','grade':'Good','health_score':98,'confidence':.8,'defects':[],'summary':'Clean.'}
    chat_seq([rusty,clean]);r=condition.assess_verified(im,'rru',(124,162))
    assert r['verified']=='disputed' and r['confidence']<=.3 and 'check visually' in r['summary'] and r['defects'][0]['confirmed'] is False
    chat_seq([rusty,{**rusty,'defects':[d('corrosion','minor',15)]}]);r=condition.assess_verified(im,'rru',(124,162))
    assert r['verified']=='confirmed' and r['defects'][0]['confirmed']  # rust vs corrosion count as the same finding
    chat_seq([clean]);r=condition.assess_verified(im,'rru',(124,162))
    assert r['grade']=='Good' and r['verified']=='not_run'  # nothing to re-check

def test_condition_saved_and_exported(monkeypatch):
    pytest.importorskip('PIL')
    from app import condition
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Condition'}).json()
        m=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('DJI_0003.JPG',_drone_jpeg(1),'image/jpeg')}).json()
        defects=[{'type':'rust','severity':'moderate','extent_percent':20,'location':'bracket base','evidence':'orange-brown flaking','source':'ai','confirmed':True}]
        boxes=[{'class_name':'mounting_brackets','x1':2,'y1':2,'x2':40,'y2':40,'condition_grade':'Poor','condition_score':55,'condition_defects':defects,'condition_source':'ai','condition_confidence':.7,'condition_verified':'confirmed','condition_summary':'Rust on bracket.'},
               {'class_name':'sector_antenna','x1':42,'y1':2,'x2':62,'y2':46,'condition_grade':'Good','condition_score':95,'condition_defects':[],'condition_source':'reviewer','condition_confidence':.9,'condition_verified':'confirmed'},
               {'class_name':'bolts','x1':5,'y1':42,'x2':9,'y2':46,'condition_grade':'Shiny'}]  # invalid grade: ignored
        a=client.put(f"/api/v1/media/{m['id']}/annotations",json={'workflow':'manual_qc','image_width':64,'image_height':48,'boxes':boxes}).json()['annotations']
        assert (a[0]['condition_grade'],a[0]['condition_verified'],a[0]['condition_defects'][0]['extent_percent'])==('Poor','confirmed',20)
        assert (a[1]['condition_source'],a[1]['condition_confidence'],a[1]['condition_verified'])==('reviewer',None,None)  # reviewer grades carry no AI confidence
        assert a[2]['condition_grade'] is None
        csv_text=client.get(f"/api/v1/inspections/{i['id']}/annotations.csv").text
        assert 'Poor,55,20,rust moderate ~20% @ bracket base,ai,confirmed,Rust on bracket.' in csv_text
        monkeypatch.setattr(condition.vendor,'status',lambda:'ready')
        monkeypatch.setattr(condition,'assess_verified',lambda im,cls,size:{'grade':'Good','size':list(size),'cls':cls})
        r=client.post(f"/api/v1/media/{m['id']}/condition-assessment",json={'class_name':'rru','x1':2,'y1':2,'x2':40,'y2':30,'image_width':64,'image_height':48}).json()
        assert r=={'grade':'Good','size':[38.0,28.0],'cls':'rru'}

def test_site_inventory_from_approved_images():
    with TestClient(app) as client:
        site=client.post('/api/v1/sites',json={'code':'TWR-INV-'+str(__import__('uuid').uuid4())[:6],'name':'Inventory test'}).json()
        i=client.post('/api/v1/inspections',json={'site_id':site['id'],'label':'Survey'}).json()
        up=lambda n:client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':(n,b'\x89PNG\r\n\x1a\n','image/png')}).json()
        m1,m2=up('a.png'),up('b.png')
        boxes=[{'class_name':'sector_antenna','x1':1,'y1':1,'x2':20,'y2':40,'manufacturer':'Ericsson','vendor_source':'reviewer','condition_grade':'Good','condition_score':95,'condition_defects':[],'condition_source':'reviewer'},
               {'class_name':'mounting_brackets','x1':22,'y1':1,'x2':40,'y2':20,'condition_grade':'Poor','condition_score':50,'condition_source':'ai','condition_verified':'confirmed',
                'condition_defects':[{'type':'rust','severity':'moderate','extent_percent':20,'location':'base','evidence':'brown'}]},
               {'class_name':'sector_antenna','x1':42,'y1':1,'x2':60,'y2':40}]
        client.put(f"/api/v1/media/{m1['id']}/annotations",json={'workflow':'manual_qc','image_width':64,'image_height':48,'boxes':boxes})
        inv=client.get(f"/api/v1/sites/{site['id']}/inventory").json()
        s=inv['summary']
        assert (s['images_total'],s['images_approved'],s['items'],s['items_with_defects'])==(2,1,3,1)
        assert s['grades']=={'Good':1,'Poor':1,'Not assessed':1} and s['manufacturers']=={'Ericsson':1} and s['defect_types']=={'rust':1} and s['verified']=={'confirmed':1}
        ant=[c for c in s['by_class'] if c['class_name']=='sector_antenna'][0]
        assert ant['count']==2 and ant['manufacturers']=={'Ericsson':1}
        imgs={x['filename']:x for x in inv['images']}
        assert imgs['a.png']['status']=='approved' and len(imgs['a.png']['items'])==3 and imgs['a.png']['image_width']==64
        assert imgs['b.png']['status']=='pending' and imgs['b.png']['items']==[]
        assert client.get(f"/api/v1/sites/{site['id']}/inventory",params={'inspection_id':999999}).json()['images']==[]
        d=client.get('/api/v1/dashboard').json();assert d['observed_items']>=3 and d['items_with_defects']>=1

def test_frame_plan_and_real_extraction(tmp_path):
    from app import frames
    assert frames.plan(1000,30,300)==list(range(0,1000,30))  # ~1 frame/s at 30 fps
    assert len(frames.plan(30000,30,300))==300 and frames.plan(30000,30,300)[1]==100  # long flight: stride widens to stay under the cap
    cv2=pytest.importorskip('cv2');np=pytest.importorskip('numpy')
    v=tmp_path/'clip.mp4';w=cv2.VideoWriter(str(v),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
    for i in range(90):w.write(np.full((48,64,3),i*2,np.uint8))
    w.release()
    out=frames.extract(v,tmp_path/'frames',stride=30)
    assert [(f,round(t,2),wd,ht) for f,t,_,wd,ht in out]==[(0,0.0,64,48),(30,1.0,64,48),(60,2.0,64,48)] and all(p.is_file() for _,_,p,_,_ in out)
    assert frames.label('DJI_0042.MP4',30,61.5)==('DJI_0042_f000030.jpg','01:01.5')

def test_video_frames_and_scoped_processing(monkeypatch):
    pytest.importorskip('PIL')
    from PIL import Image
    from app import detector, frames
    from app.main import ROOT
    box={'class_id':0,'class_name':'sector_antenna','confidence':.9,'x1':1,'y1':1,'x2':20,'y2':30}
    monkeypatch.setattr(detector,'status',lambda:'ready')
    monkeypatch.setattr(detector,'detect',lambda path,video=False:iter([(0,64,48,[box])]))
    calls=[]
    def fake_extract(video_path,out_dir,*a,**k):
        calls.append(1);out_dir.mkdir(parents=True,exist_ok=True);out=[]
        for idx,sec in ((0,0.0),(30,1.0)):
            p=out_dir/f'{idx:06d}.jpg';Image.new('RGB',(64,48),'gray').save(p);out.append((idx,sec,p,64,48))
        return out
    monkeypatch.setattr(frames,'extract',fake_extract)
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Video'}).json()
        img=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('a.jpg',_drone_jpeg(1),'image/jpeg')}).json()
        vid=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('DJI_0042.MP4',b'not really a video','video/mp4')}).json()
        j=client.post(f"/api/v1/inspections/{i['id']}/process",params={'media_ids':[vid['id']]}).json()  # just the new upload
        assert j['media_ids']==[vid['id']]
        j=client.get(f"/api/v1/jobs/{j['id']}").json()
        assert j['status']=='completed' and 'incl. 2 video frames' in j['detail'],j
        media=client.get(f"/api/v1/inspections/{i['id']}/media").json()
        fr=[m for m in media if m['parent_media_id']==vid['id']]
        assert [(m['filename'],m['frame_index'],m['frame_time'],m['content_type']) for m in fr]==[('DJI_0042_f000000.jpg',0,0.0,'image/jpeg'),('DJI_0042_f000030.jpg',30,1.0,'image/jpeg')]
        assert client.get(fr[1]['url']).status_code==200  # served like any upload
        dets=client.get(f"/api/v1/inspections/{i['id']}/detections").json()
        assert sorted(d['media_id'] for d in dets)==sorted(m['id'] for m in fr)  # frames have boxes; the image wasn't processed yet
        j2=client.post(f"/api/v1/inspections/{i['id']}/process",params={'only_new':True}).json()
        assert j2['media_ids']==[img['id']]  # only the image is new
        dets=client.get(f"/api/v1/inspections/{i['id']}/detections").json()
        assert sorted(d['media_id'] for d in dets)==sorted([img['id']]+[m['id'] for m in fr])  # each file from its latest run
        assert client.post(f"/api/v1/inspections/{i['id']}/process",params={'only_new':True}).status_code==409
        client.post(f"/api/v1/inspections/{i['id']}/process")  # full re-run reuses the frames
        assert len(calls)==1 and len([m for m in client.get(f"/api/v1/inspections/{i['id']}/media").json() if m['parent_media_id']])==2
        assert len(client.get(f"/api/v1/inspections/{i['id']}/detections").json())==3
        site=client.get(f"/api/v1/sites/1/inventory",params={'inspection_id':i['id']}).json()
        assert {x['filename']:x['video'] for x in site['images']}=={'a.jpg':None,'DJI_0042_f000000.jpg':'DJI_0042.MP4','DJI_0042_f000030.jpg':'DJI_0042.MP4'}

def test_annotated_output_uses_each_files_latest_run(monkeypatch):
    pytest.importorskip('PIL')
    from app import detector
    box={'class_id':0,'class_name':'sector_antenna','confidence':.9,'x1':2,'y1':2,'x2':30,'y2':40}
    monkeypatch.setattr(detector,'status',lambda:'ready')
    monkeypatch.setattr(detector,'detect',lambda path,video=False:iter([(0,64,48,[box])]))
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Scoped runs'}).json()
        old=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('old.jpg',_drone_jpeg(1),'image/jpeg')}).json()
        client.post(f"/api/v1/inspections/{i['id']}/process")
        new=client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':('new.jpg',_drone_jpeg(1),'image/jpeg')}).json()
        client.post(f"/api/v1/inspections/{i['id']}/process",params={'only_new':True})  # newest run covers only new.jpg
        for m in (old,new):  # the older photo still downloads with its boxes from the earlier run
            r=client.get(f"/api/v1/media/{m['id']}/annotated?source=model")
            assert r.status_code==200 and r.headers['x-box-count']=='1',(m['filename'],r.headers.get('x-box-count'))

def test_review_regressions(monkeypatch):
    pytest.importorskip('PIL')
    import urllib.request
    from PIL import Image
    from app import detector, frames, vendor, condition
    box={'class_id':0,'class_name':'sector_antenna','confidence':.9,'x1':2,'y1':2,'x2':30,'y2':40}
    monkeypatch.setattr(detector,'status',lambda:'ready')
    monkeypatch.setattr(detector,'detect',lambda path,video=False:iter([(0,64,48,[box])]))
    def bad_video(*a,**k):raise RuntimeError('Cannot open video broken.MOV')
    monkeypatch.setattr(frames,'extract',bad_video)
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Regressions'}).json()
        up=lambda n,b,t:client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':(n,b,t)}).json()
        a=up('a.jpg',_drone_jpeg(1),'image/jpeg');v=up('broken.MOV',b'xx','video/quicktime')
        j=client.post(f"/api/v1/inspections/{i['id']}/process").json();j=client.get(f"/api/v1/jobs/{j['id']}").json()
        assert j['status']=='completed' and 'FAILED 1: broken.MOV' in j['detail'] and j['media_ids']==[a['id']]  # bad video reported, photo kept
        assert len(client.get(f"/api/v1/inspections/{i['id']}/detections").json())==1
        b=up('b.jpg',_drone_jpeg(1),'image/jpeg')
        j=client.post(f"/api/v1/inspections/{i['id']}/process",params={'only_new':True}).json()
        assert sorted(j['media_ids'])==sorted([v['id'],b['id']])  # uploaded after a full run = new; the failed video is retried
        # an HTML file declared as an image is stored as .png and can't be rendered as a page
        h=up('x.html',b'<script>alert(1)</script>','image/png')
        assert h['storage_key'].endswith('.png')
        assert h['storage_key'].startswith(f"sites/TWR-001/inspections/INS-{i['id']:06d}/images/")
        from app import media_paths,storage
        from app.tasks import ROOT
        assert storage.key_for(ROOT/h['storage_key'])=='uploads/'+h['storage_key']  # identical hierarchy in production S3
        assert media_paths.frame_dir(h['storage_key'],123).as_posix()==f"sites/TWR-001/inspections/INS-{i['id']:06d}/frames/MEDIA-000123"
        r=client.get('/uploads/'+h['storage_key']);assert r.headers['x-content-type-options']=='nosniff' and r.headers['content-type'].startswith('image/png')
    def timeout(*a,**k):raise TimeoutError('timed out')
    monkeypatch.setattr(urllib.request,'urlopen',timeout)
    with pytest.raises(RuntimeError,match='timed out'):vendor._chat('p','img',{})
    monkeypatch.setattr(vendor,'compare_model',lambda:'thinking-model')
    first={'grade':'Fair','health_score':80,'defects':[{'type':'rust','severity':'minor','extent_percent':10,'location':'x','evidence':'y','source':'ai'}],'rust_percent':10,'image_quality':'good','confidence':.7,'summary':'s','recommended_action':'','model':'m','seconds':1}
    r=condition.verify(Image.new('RGB',(80,80)),'rru',(80,80),first)
    assert r['verified']=='not_run' and r['grade']=='Fair'  # a timed-out second opinion keeps the first assessment

def test_ply_roundtrip_and_orientation(tmp_path):
    np=pytest.importorskip('numpy')
    from app import reconstruct as rc
    pts=np.random.default_rng(1).normal(size=(50,3)).astype(np.float32);cols=np.random.default_rng(2).integers(0,255,(50,3),dtype=np.uint8);faces=np.array([[0,1,2],[2,3,4]],np.int32)
    rc.write_ply(tmp_path/'m.ply',pts,cols,faces);r=rc.read_ply(tmp_path/'m.ply')
    assert np.allclose(r['points'],pts) and (r['colors']==cols).all() and (r['faces']==faces).all()
    (tmp_path/'a.ply').write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n4 5 6\n')
    assert rc.read_ply(tmp_path/'a.ply')['points'].tolist()==[[1,2,3],[4,5,6]]
    # synthetic drone orbit: 12 cameras circling 6 units above a vertical "tower", looking at it; then the whole scene is rotated arbitrarily
    th=np.linspace(0,2*np.pi,12,endpoint=False);C=np.stack([10*np.cos(th),10*np.sin(th),np.full(12,6.0)],1)
    tower=np.stack([np.random.default_rng(3).normal(0,.3,400),np.random.default_rng(4).normal(0,.3,400),np.random.default_rng(5).uniform(-8,4,400)],1)
    a,b=.7,-1.1;Rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]);Rz=np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1]]);R0=Rz@Rx
    lines=['# images']
    for i,c in enumerate(C):
        z=-c/np.linalg.norm(c);x=np.cross(z,[0,0,1.0]);x/=np.linalg.norm(x);y=np.cross(z,x);R=np.stack([x,y,z])@R0.T  # world->camera in the rotated world
        cw=R0@c;t=-R@cw;tr=np.trace(R);qw=np.sqrt(max(0,1+tr))/2
        qx,qy,qz=(R[2,1]-R[1,2])/(4*qw),(R[0,2]-R[2,0])/(4*qw),(R[1,0]-R[0,1])/(4*qw)
        lines+=[f'{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 img{i}.jpg','']
    (tmp_path/'images.txt').write_text('\n'.join(lines))
    cams=rc.read_cameras_txt(tmp_path/'images.txt')
    assert np.allclose(cams[0]['center'],R0@C[0],atol=1e-6)  # centres recovered from the quaternions
    M=rc.normalising_transform((tower@R0.T).astype(np.float32),cams)
    up=(R0@[0,0,1.0])@M[:3,:3].T;up/=np.linalg.norm(up)
    assert up[1]>.98  # the tower stands upright (+Y)
    assert all(rc._apply(M,np.array([c['center']]))[0][1]>0 for c in cams)  # cameras above the scene centre

def test_reconstruction_job_flow(monkeypatch,tmp_path):
    np=pytest.importorskip('numpy')
    from app import reconstruct as rc
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'3D'}).json()
        assert client.post(f"/api/v1/inspections/{i['id']}/reconstruct").status_code==503  # COLMAP not configured in tests
        monkeypatch.setattr(rc,'status',lambda:'ready')
        up=lambda n:client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':(n,_drone_jpeg(1),'image/jpeg')}).json()
        up('DJI_0002.JPG');up('DJI_0001.JPG')
        assert client.post(f"/api/v1/inspections/{i['id']}/reconstruct").status_code==409  # fewer than 3 photos
        up('DJI_0003.JPG');seen={}
        def fake_run(work,photos,videos,dense,progress):
            seen.update(names=[p.name for p in photos],dense=dense);progress('mapping',40,'Solving cameras')
            (work/'web').mkdir(parents=True);rc.write_ply(work/'web'/'points.ply',np.zeros((3,3),np.float32));(work/'web'/'cameras.json').write_text('[]')
            return {'registered_images':3,'images_total':3,'sparse_points':1234,'models_found':1,'dense':dense,'dense_points':99,'mesh_faces':10,'seconds':5,
                    'files':{'web_points':'web/points.ply','web_cameras':'web/cameras.json','log':'colmap.log'}}
        monkeypatch.setattr(rc,'run',fake_run)
        j=client.post(f"/api/v1/inspections/{i['id']}/reconstruct",params={'dense':True}).json()
        assert j['kind']=='reconstruction'
        assert len(seen['names'])==3 and seen['dense'] is True
        job=client.get(f"/api/v1/jobs/{j['id']}").json()
        assert job['status']=='completed' and '3/3 images placed' in job['detail'] and '99 dense points' in job['detail']
        ms=client.get(f"/api/v1/inspections/{i['id']}/models").json()
        assert ms[0]['status']=='completed' and ms[0]['dense'] is True and ms[0]['urls']['web_points'].endswith('/web/points.ply')
        assert client.get(ms[0]['urls']['web_points']).status_code==200 and client.get(f"/api/v1/inspections/{i['id']}/model").json()['id']==ms[0]['id']
        def boom(*a,**k):raise rc.Failed('COLMAP could not connect the images into one model: they need more overlap')
        monkeypatch.setattr(rc,'run',boom)
        j=client.post(f"/api/v1/inspections/{i['id']}/reconstruct").json()
        job=client.get(f"/api/v1/jobs/{j['id']}").json();m=client.get(f"/api/v1/inspections/{i['id']}/models").json()[0]
        assert job['status']=='failed' and 'more overlap' in job['detail'] and m['status']=='failed' and m['urls']['log'].endswith('colmap.log')
        assert client.get(f"/api/v1/inspections/{i['id']}/detections").json()==[]  # reconstruction jobs aren't detection runs

def _fake_colmap(rc,np):
    """Stand-in for colmap.exe that writes the files each command would produce."""
    calls=[]
    def colmap(args,log,cwd=None):
        cmd=args[0];a={str(args[i]):args[i+1] for i in range(1,len(args)-1,2)};calls.append(cmd)
        with open(log,'a') as f:f.write(f'$ colmap {cmd}\n')
        if cmd=='feature_extractor':__import__('pathlib').Path(a['--database_path']).write_bytes(b'x'*1000)
        elif cmd=='mapper':  # a small text model (the app reads models itself, without colmap.exe)
            d=__import__('pathlib').Path(a['--output_path'])/'0';d.mkdir(parents=True,exist_ok=True)
            (d/'cameras.txt').write_text('1 SIMPLE_PINHOLE 64 48 50 32 24\n');(d/'points3D.txt').write_text('1 0 0 0 1 1 1 0.5\n')
            (d/'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 {i} 0 5 1 photos/{i:05d}.jpg\n\n' for i in range(3)))
        elif cmd=='model_analyzer':
            with open(log,'a') as f:f.write('Registered images: 3\nPoints: 500\nMean track length: 4.2\nMean reprojection error: 0.8px\n')
        elif cmd=='model_converter' and a['--output_type']=='PLY':rc.write_ply(a['--output_path'],np.random.default_rng(0).normal(size=(500,3)).astype(np.float32))
        elif cmd=='model_converter':
            p=__import__('pathlib').Path(a['--output_path']);p.mkdir(parents=True,exist_ok=True)
            (p/'images.txt').write_text('\n'.join(f'{i+1} 1 0 0 0 {i} 0 5 1 photos/{i:05d}.jpg\n' for i in range(3)))
        elif cmd=='image_undistorter':(__import__('pathlib').Path(a['--output_path'])/'images').mkdir(parents=True,exist_ok=True)
        elif cmd=='patch_match_stereo':
            s=__import__('pathlib').Path(a['--workspace_path'])/'stereo'/'depth_maps';s.mkdir(parents=True,exist_ok=True);(s/'x.bin').write_bytes(b'd'*5000)
        elif cmd=='stereo_fusion':rc.write_ply(a['--output_path'],np.random.default_rng(1).normal(size=(2000,3)).astype(np.float32),np.full((2000,3),200,np.uint8))
        elif cmd=='poisson_mesher':rc.write_ply(a['--output_path'],np.eye(3,dtype=np.float32),faces=np.array([[0,1,2]],np.int32))
    return colmap,calls

def test_sparse_then_densify_pipeline(monkeypatch,tmp_path):
    np=pytest.importorskip('numpy');pytest.importorskip('PIL')
    import os
    from PIL import Image
    from app import reconstruct as rc
    fake,calls=_fake_colmap(rc,np);monkeypatch.setattr(rc,'colmap',fake)
    photos=[]
    for i in range(3):p=tmp_path/f'DJI_{i}.JPG';Image.new('RGB',(64,48)).save(p);photos.append(p)
    steps=[];prog=lambda s,p,d:steps.append((s,p))
    r=rc.run(tmp_path/'w1',photos,[],False,prog)
    assert (r['registered_images'],r['sparse_points'],r['sparse_model'],r['dense'])==(3,500,'sparse/0',False) and len(r['transform'])==4
    w1=tmp_path/'w1'
    assert not (w1/'database.db').exists() and not (w1/'sparse_txt').exists()  # build-only files cleaned up
    assert (w1/'sparse'/'0'/'images.txt').exists() and (w1/'web'/'points.ply').exists() and (w1/'web'/'cameras.json').exists()
    linked=w1/'images'/'photos'/'00000.jpg'
    assert linked.exists() and (os.stat(linked).st_nlink>=2 or os.path.samefile(linked,photos[0]) or linked.stat().st_size==photos[0].stat().st_size)  # hard link where possible
    calls.clear()
    d=rc.densify(w1,tmp_path/'w2',prog,r)
    assert 'feature_extractor' not in calls and 'mapper' not in calls and calls.count('patch_match_stereo')==1  # reuses the sparse solution
    w2=tmp_path/'w2'
    assert d['dense'] and d['dense_points']==2000 and d['mesh_faces']==1 and d['transform']==r['transform'] and d['registered_images']==3
    assert not (w2/'dense'/'stereo').exists() and (w2/'dense'/'fused.ply').exists() and (w2/'web'/'mesh.ply').exists() and (w2/'sparse.ply').exists()
    (w1/'sparse'/'0'/'images.txt').unlink()
    with pytest.raises(rc.Failed,match='missing'):rc.densify(w1,tmp_path/'w3',prog,{})  # old build without its sparse model

def test_densify_endpoint(monkeypatch):
    np=pytest.importorskip('numpy')
    from app import reconstruct as rc
    monkeypatch.setattr(rc,'status',lambda:'ready')
    def fake_run(work,photos,videos,dense,progress):
        (work/'web').mkdir(parents=True);rc.write_ply(work/'web'/'points.ply',np.zeros((3,3),np.float32))
        return {'registered_images':3,'images_total':3,'sparse_points':10,'models_found':1,'dense':False,'seconds':1,'sparse_model':'sparse/0','transform':np.eye(4).tolist(),'files':{'web_points':'web/points.ply'}}
    got={}
    def fake_densify(src,work,progress,q):
        got.update(src=src.name,q=q);(work/'web').mkdir(parents=True);rc.write_ply(work/'web'/'points.ply',np.zeros((3,3),np.float32))
        return {**{k:q[k] for k in ('registered_images','images_total','sparse_points','models_found')},'dense':True,'dense_points':99,'mesh_faces':5,'seconds':2,'files':{'web_points':'web/points.ply','web_mesh':'web/points.ply'}}
    monkeypatch.setattr(rc,'run',fake_run);monkeypatch.setattr(rc,'densify',fake_densify)
    with TestClient(app) as client:
        i=client.post('/api/v1/inspections',json={'site_id':1,'label':'Densify'}).json()
        for n in ('a.jpg','b.jpg','c.jpg'):client.post(f"/api/v1/inspections/{i['id']}/media",files={'file':(n,_drone_jpeg(1),'image/jpeg')})
        sp=client.post(f"/api/v1/inspections/{i['id']}/reconstruct").json()
        src=client.get(f"/api/v1/inspections/{i['id']}/models").json()[0]
        j=client.post(f"/api/v1/models/{src['id']}/densify").json()
        assert got['src']==f"{i['id']}_{sp['id']}" and got['q']['sparse_model']=='sparse/0'
        job=client.get(f"/api/v1/jobs/{j['id']}").json();m=client.get(f"/api/v1/inspections/{i['id']}/models").json()[0]
        assert job['status']=='completed' and f"dense from model #{src['id']}" in job['detail'] and m['dense'] and m['quality']['source_model']==src['id'] and m['urls']['web_mesh']
        assert client.post(f"/api/v1/models/{m['id']}/densify").status_code==409  # already dense
        assert client.post('/api/v1/models/999999/densify').status_code==404

def test_registration_boxes_from_synthetic_colmap(monkeypatch,tmp_path):
    np=pytest.importorskip('numpy')
    from types import SimpleNamespace as NS
    from app import registration as reg
    rng=np.random.default_rng(0);f,W,H=1000.0,2000,1500
    # two 0.6 wide x 1.2 tall x 0.2 deep panel antennas facing outward on opposite sides of a tower at the origin
    ants={'east':np.array([2.0,3.0,0.0]),'west':np.array([-2.0,3.0,0.0])}
    pts=[];pid=1;owner={}
    for name,c in ants.items():
        for _ in range(150):
            p=c+[rng.uniform(-.1,.1),rng.uniform(-.6,.6),rng.uniform(-.3,.3)];pts.append((pid,p));owner[pid]=name;pid+=1
    def look(cam,target):
        z=target-cam;z/=np.linalg.norm(z);x=np.cross(z,[0,1.0,0]);x/=np.linalg.norm(x);y=np.cross(z,x);return np.stack([x,y,z])  # COLMAP: x right, y down, z forward
    def quat(R):
        qw=np.sqrt(max(0,1+np.trace(R)))/2;return qw,(R[2,1]-R[1,2])/(4*qw),(R[0,2]-R[2,0])/(4*qw),(R[1,0]-R[0,1])/(4*qw)
    cams=[np.array([6.0*np.cos(a),5.0,6.0*np.sin(a)]) for a in np.radians([10,80,170,260])]
    img_lines=[];dets=[];media=[]
    for i,C in enumerate(cams):
        R=look(C,np.array([0,3.0,0]));t=-R@C;obs=[];per={}
        for pid_,p in pts:
            q=R@p+t
            if q[2]<=0:continue
            u,v=f*q[0]/q[2]+W/2,f*q[1]/q[2]+H/2
            if 0<=u<W and 0<=v<H:obs.append(f'{u} {v} {pid_}');per.setdefault(owner[pid_],[]).append((u,v))
        qw,qx,qy,qz=quat(R);img_lines+=[f'{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 photos/{i:05d}.jpg',' '.join(obs)]
        media.append(NS(id=100+i,parent_media_id=None,content_type='image/jpeg',filename=f'DJI_{i:04d}.JPG',storage_key=f'k{i}.jpg'))
        for name,uv in per.items():
            uv=np.array(uv);dets.append(NS(id=len(dets)+1,media_id=100+i,class_name='sector_antenna',confidence=.9,x1=uv[:,0].min(),y1=uv[:,1].min(),x2=uv[:,0].max(),y2=uv[:,1].max()))
    txt=tmp_path/'txt';txt.mkdir();(tmp_path/'sparse'/'0').mkdir(parents=True)
    (txt/'images.txt').write_text('\n'.join(img_lines)+'\n');(txt/'cameras.txt').write_text(f'1 PINHOLE {W} {H} {f} {f} {W/2} {H/2}\n')
    (txt/'points3D.txt').write_text('\n'.join(f'{i} {p[0]} {p[1]} {p[2]} 128 128 128 0.5' for i,p in pts)+'\n')
    def fake_convert(model_dir):yield txt
    monkeypatch.setattr(reg,'_convert_model',fake_convert)
    out=reg.build_candidates(tmp_path,np.eye(4).tolist(),media,dets)
    assert len(out)==2 and {c['class_name'] for c in out}=={'sector_antenna'}
    for c in out:
        side='east' if c['position'][0]>0 else 'west';centre=ants[side]
        assert np.linalg.norm(c['position']-centre)<.25  # where the antenna is
        assert .4<c['size_w']<.9 and .9<c['size_h']<1.6 and c['size_d']<c['size_w']  # about 0.6 wide, 1.2 tall, thin
        assert abs(abs(c['yaw'])-np.pi/2)<.3  # faces outward along +/-X, away from the tower axis
    assert reg.rule_for('bolts')['radius']<reg.rule_for('sector_antenna')['radius'] and reg.rule_for('anything_new')==reg.DEFAULT_RULE  # every class is registered

def test_registration_dense_boxes_separate_neighbours(monkeypatch,tmp_path):
    np=pytest.importorskip('numpy')
    from types import SimpleNamespace as NS
    from app import registration as reg, reconstruct
    rng=np.random.default_rng(1);f,W,H=1200.0,2000,1500
    # two 0.3 wide x 1.2 tall x 0.15 deep panels side by side (0.45 apart) on the east face of a pole at the origin;
    # the sparse model has almost nothing on them, the dense cloud covers their surfaces and the pole behind
    ants=[np.array([1.0,3.0,-.225]),np.array([1.0,3.0,.225])];half=np.array([.075,.6,.15])
    def surface(c,n):
        p=rng.uniform(-1,1,(n,3));k=rng.integers(0,3,n);p[np.arange(n),k]=np.sign(p[np.arange(n),k]);return c+p*half
    pole=np.c_[.3*np.cos(a:=rng.uniform(0,2*np.pi,20000)),rng.uniform(0,6,20000),.3*np.sin(a)]
    dense=np.vstack([surface(c,6000) for c in ants]+[pole])
    reconstruct.write_ply(tmp_path/'fused.ply',dense.astype(np.float32))
    def look(cam,target):
        z=target-cam;z/=np.linalg.norm(z);x=np.cross(z,[0,1.0,0]);x/=np.linalg.norm(x);y=np.cross(z,x);return np.stack([x,y,z])
    def quat(R):
        qw=np.sqrt(max(0,1+np.trace(R)))/2;return qw,(R[2,1]-R[1,2])/(4*qw),(R[0,2]-R[2,0])/(4*qw),(R[1,0]-R[0,1])/(4*qw)
    img_lines=[];dets=[];media=[];sparse=[]
    corners=lambda c:c+np.array([[sx,sy,sz] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)])*half
    for i,a in enumerate(np.radians([-35,-15,0,15,35])):
        C=np.array([6*np.cos(a),4.0,6*np.sin(a)]);R=look(C,np.array([1.0,3.0,0]));t=-R@C
        qw,qx,qy,qz=quat(R);img_lines+=[f'{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 photos/{i:05d}.jpg','']
        media.append(NS(id=100+i,parent_media_id=None,content_type='image/jpeg',filename=f'DJI_{i:04d}.JPG',storage_key=f'k{i}.jpg'))
        for c in ants:  # YOLO box = the panel's outline in this photo
            q=corners(c)@R.T+t;uv=f*q[:,:2]/q[:,2:]+[W/2,H/2]
            dets.append(NS(id=len(dets)+1,media_id=100+i,class_name='sector_antenna',confidence=.9,image_width=W,image_height=H,
                           x1=uv[:,0].min(),y1=uv[:,1].min(),x2=uv[:,0].max(),y2=uv[:,1].max()))
    txt=tmp_path/'txt';txt.mkdir();(tmp_path/'sparse'/'0').mkdir(parents=True)
    (txt/'images.txt').write_text('\n'.join(img_lines)+'\n');(txt/'cameras.txt').write_text(f'1 PINHOLE {W} {H} {f} {f} {W/2} {H/2}\n')
    (txt/'points3D.txt').write_text('1 0 3 0 128 128 128 0.5\n')
    def fake_convert(model_dir):yield txt
    monkeypatch.setattr(reg,'_convert_model',fake_convert)
    out=reg.build_candidates(tmp_path,np.eye(4).tolist(),media,dets,tmp_path/'fused.ply')
    assert len(out)==2 and all(c['box_method']=='dense' and c['media_count']==5 for c in out)  # one box per panel, not merged
    for c in sorted(out,key=lambda c:c['position'][2]):
        truth=ants[0] if c['position'][2]<0 else ants[1]
        assert np.linalg.norm(c['position']-truth)<.1
        assert .2<c['size_w']<.45 and 1.0<c['size_h']<1.4 and c['size_d']<c['size_w']  # 0.3 wide, 1.2 tall, thin
        assert abs(abs(c['yaw'])-np.pi/2)<.35  # thin side along X (both panels sit on one face, so the tower axis can't say which way is out)
    # a group whose boxes don't line up with each other in 3D is not one object: dropped, not drawn
    assert reg.fit_box(np.zeros((2,3)),np.zeros(2)) is None and reg.fit_box(np.full((10,3),np.nan),np.zeros(2)) is None

def test_mounting_bracket_refinement_can_keep_dense_surface_anchor():
    np=pytest.importorskip('numpy')
    from types import SimpleNamespace as NS
    from app import registration as reg
    cam=('PINHOLE',[1000.,1000.,1000.,750.],2000,1500)
    pose=(None,None,None,1,np.eye(3),np.zeros(3))
    poses={'a':pose,'b':pose};cameras={1:cam}
    # The photo boxes are deliberately offset from the dense box centre. Structural brackets must stay on
    # the reconstructed surface instead of being pulled toward the triangulated centres of irregular 2D boxes.
    det=lambda: NS(x1=1080.,x2=1320.,y1=650.,y2=850.,image_width=2000,image_height=1500)
    candidate={'class_name':'mounting_brackets','position':np.array([0.,0.,5.]),'size_w':1.,'size_h':1.,'size_d':.25,'yaw':0.}
    reg.fit_to_detections(candidate,[('a',det()),('b',det())],poses,cameras,np.eye(4),rounds=0,move_centre=False)
    assert np.allclose(candidate['position'],[0.,0.,5.])
    assert 'mounting_brackets' in reg.SURFACE_ANCHORED_CLASSES

def test_login_sessions_and_role_enforcement(monkeypatch):
    monkeypatch.setenv('AUTH_DISABLED','false')
    with TestClient(app) as client:
        viewer=client.post('/api/v1/auth/login',json={'email':'viewer@aeromind.local','password':'AeroMind@123'});assert viewer.status_code==200
        vh={'Authorization':'Bearer '+viewer.json()['token']}
        assert client.get('/api/v1/auth/me',headers=vh).json()['role']=='viewer'
        assert client.get('/api/v1/sites',headers=vh).status_code==200
        assert client.post('/api/v1/sites',headers=vh,json={'code':'NOPE','name':'Not allowed'}).status_code==403
        admin=client.post('/api/v1/auth/login',json={'email':'admin@aeromind.local','password':'AeroMind@123'});assert admin.status_code==200
        ah={'Authorization':'Bearer '+admin.json()['token']}
        assert client.get('/api/v1/users',headers=ah).status_code==200
        made=client.post('/api/v1/users',headers=ah,json={'email':'new@example.com','name':'New Inspector','role':'inspector','password':'SafePassword1!'});assert made.status_code==201
        assert client.post('/api/v1/auth/logout',headers=vh).status_code==204
        assert client.get('/api/v1/auth/me',headers=vh).status_code==401

def test_remote_gpu_worker_runs_queued_work(monkeypatch):
    """GPU_WORKER=remote (Railway): GPU work is queued; the worker claims it over HTTPS with its token, computes it
    without database access and posts the result, which the server applies. Errors come back to the waiting request."""
    import threading
    from app import gpu, worker, tasks
    from app.database import SessionLocal
    from app.models import WorkerTask, Job, Detection
    monkeypatch.setattr(gpu, 'REMOTE', True); monkeypatch.setattr(gpu, 'TOKEN', 'tok'); monkeypatch.setattr(gpu, 'CALL_TIMEOUT', 20)
    caps = {'detector': {'status': 'ready', 'model': 'w/best.pt', 'classes': {'0': 'rru'}}, 'reconstruction': {'status': 'ready'}, 'gpu': 'Test GPU'}
    monkeypatch.setattr(worker, 'capabilities', lambda: caps)
    monkeypatch.setattr(tasks.detector, 'detect', lambda path, save: [(None, 640, 480, [{'class_id': 0, 'class_name': 'rru', 'confidence': .9, 'x1': 1, 'y1': 2, 'x2': 30, 'y2': 40}])])
    stop = threading.Event(); monkeypatch.setattr(worker, '_stop', stop)
    with TestClient(app) as client:
        assert client.post('/api/v1/worker/claim', json={'name': 'x'}).status_code == 401  # token required
        r = client.post('/api/v1/models/999999/register-assets')  # nobody connected yet: fail fast, say why
        assert r.status_code == 503 and 'offline' in r.json()['detail']
        wc = TestClient(app, headers={'X-Worker-Token': 'tok'})
        assert wc.post('/api/v1/worker/claim', json={'name': 'test-gpu', 'info': caps}).status_code == 204  # heartbeat, nothing queued
        assert client.get('/health').json()['worker']['online'] and client.get('/api/v1/detector').json()['status'] == 'ready'
        assert {'id': 0, 'name': 'rru'} in client.get('/api/v1/classes').json()  # class list reported by the worker
        lanes = [threading.Thread(target=worker.lane_loop, args=(l, wc), daemon=True) for l in ('quick', 'long')]
        for t in lanes: t.start()
        try:
            r = client.post('/api/v1/models/999999/register-assets')  # the server-side error reaches the browser
            assert r.status_code == 404 and r.json()['detail'] == 'Completed 3D model not found'
            r = client.post('/api/v1/media/999999/condition-assessment', json={'class_name': 'rru', 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10, 'image_width': 100, 'image_height': 100})
            assert r.status_code == 404
            ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'remote'}).json()
            up = client.post(f"/api/v1/inspections/{ins['id']}/media", files={'file': ('a.jpg', b'\xff\xd8\xff\xd9', 'image/jpeg')}).json()
            job = client.post(f"/api/v1/inspections/{ins['id']}/process").json()
            assert job['status'] == 'queued' and 'w/best.pt' in job['detail']
            for _ in range(100):  # the long lane picks it up, detects on the GPU side, the server stores the boxes
                if client.get(f"/api/v1/jobs/{job['id']}").json()['status'] in ('completed', 'failed'): break
                time.sleep(.1)
            done = client.get(f"/api/v1/jobs/{job['id']}").json()
            assert done['status'] == 'completed' and done['progress'] == 100, done
            with SessionLocal() as db:
                d = db.query(Detection).filter_by(job_id=job['id']).one()
                assert (d.media_id, d.class_name, d.image_width, d.model_name) == (up['id'], 'rru', 640, 'w/best.pt')
                assert db.query(WorkerTask).filter_by(kind='run_detection').order_by(WorkerTask.id.desc()).first().status == 'done'
        finally:
            stop.set()
            for t in lanes: t.join(5)

def test_web_app_fallback_and_api_isolation(tmp_path):
    from app import main
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        if (main.DIST / 'index.html').is_file():  # built frontend: client-side routes get the app, unknown API paths stay 404
            r = client.get('/twin')
            assert r.status_code == 200 and '<div id="root"' in r.text
            assert client.get('/api/v1/nope').status_code == 404
            assert client.get('/../backend/.env').status_code in (200, 404) and 'VLM' not in client.get('/..%2fbackend%2f.env').text

def test_small_parts_may_use_photo_boxes_when_consistent():
    from app.registration import candidate_is_valid
    base = {'size_w': .002, 'size_h': .002, 'size_d': .002, 'confidence': .6, 'media_count': 6, 'fit_iou': .6, 'box_method': 'estimate'}
    assert candidate_is_valid({**base, 'class_name': 'bolts'})                      # too small for the dense cloud, but lines up in 6 photos
    assert not candidate_is_valid({**base, 'class_name': 'bolts', 'fit_iou': .2})   # detections of different bolts grouped together
    assert not candidate_is_valid({**base, 'class_name': 'bolts', 'media_count': 3})
    assert not candidate_is_valid({**base, 'class_name': 'rru', 'size_w': .04, 'size_h': .04})  # big parts still need the dense fit

def test_site_management():
    with TestClient(app) as client:
        r = client.post('/api/v1/sites', json={'code': ' twr-900 ', 'name': 'Hill Top', 'region': 'Pune', 'tower_type': 'Monopole',
                                               'height_m': 30, 'latitude': 18.5, 'longitude': 73.8, 'operator': 'Indus', 'address': '', 'notes': ' '})
        assert r.status_code == 201, r.text
        s = r.json(); assert s['code'] == 'TWR-900' and s['address'] is None and s['notes'] is None and s['status'] == 'active'
        assert client.post('/api/v1/sites', json={'code': 'TWR-900', 'name': 'Dup'}).status_code == 409   # IDs are unique, any case
        assert client.post('/api/v1/sites', json={'code': 'TWR-901', 'name': 'Bad', 'latitude': 123}).status_code == 422
        r = client.patch(f"/api/v1/sites/{s['id']}", json={'name': 'Hill Top North', 'status': 'maintenance', 'height_m': 32.5})
        assert r.status_code == 200 and r.json()['name'] == 'Hill Top North' and r.json()['status'] == 'maintenance'
        assert client.patch(f"/api/v1/sites/{s['id']}", json={'status': 'gone'}).status_code == 422
        assert client.patch(f"/api/v1/sites/{s['id']}", json={'code': 'TWR-001'}).status_code == 409   # demo site's ID
        ov = {x['id']: x for x in client.get('/api/v1/sites-overview').json()}
        assert ov[s['id']]['stats']['inspections'] == 0 and 'boxes_3d' in ov[s['id']]['stats']
        ins = client.post('/api/v1/inspections', json={'site_id': s['id'], 'label': 'Survey A'}).json()
        client.post(f"/api/v1/inspections/{ins['id']}/media", files={'file': ('a.jpg', b'\xff\xd8\xff\xd9', 'image/jpeg')})
        ov = {x['id']: x for x in client.get('/api/v1/sites-overview').json()}
        assert ov[s['id']]['stats']['inspections'] == 1 and ov[s['id']]['stats']['photos'] == 1
        summary = client.get(f"/api/v1/sites/{s['id']}/inspections-summary").json()
        assert summary[0]['label'] == 'Survey A' and summary[0]['photos'] == 1 and summary[0]['models'] == []
        assert client.delete(f"/api/v1/sites/{s['id']}").status_code == 409   # has an inspection: archive instead
        empty = client.post('/api/v1/sites', json={'code': 'TWR-902', 'name': 'Empty'}).json()
        assert client.delete(f"/api/v1/sites/{empty['id']}").status_code == 204 and client.get(f"/api/v1/sites/{empty['id']}").status_code == 404

def _synthetic_flight(scale_m=7.5, n=12):
    """Camera centres in arbitrary model units and the matching drone positions (lat/lon/alt) for a known scale."""
    import math, numpy as np
    rng = np.random.default_rng(3); a = .7
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])  # any rotation of the model frame
    enu = np.c_[20 * np.cos(np.linspace(0, 5, n)), 20 * np.sin(np.linspace(0, 5, n)), 30 + rng.uniform(-3, 3, n)]  # an orbit, metres
    centres = ((enu - enu.mean(0)) / scale_m) @ R.T + [1.0, -2.0, .5]
    lat0, lon0 = 53.538, -2.3027
    pos = [{'lat': lat0 + math.degrees(e[1] / 6378137.0), 'lon': lon0 + math.degrees(e[0] / (6378137.0 * math.cos(math.radians(lat0)))), 'alt': float(e[2]), 'rtk': True} for e in enu]
    return [{'name': f'photos/{i:05d}.jpg', 'center': c.tolist()} for i, c in enumerate(centres)], pos

def test_scale_from_drone_positions(tmp_path):
    from app import scale
    cams, pos = _synthetic_flight()
    s = scale.estimate(cams, {c['name']: p for c, p in zip(cams, pos)})
    assert abs(s['m_per_unit'] - 7.5) < .01 and s['method'] == 'gps_rtk' and s['residual_median_m'] < .02 and s['cameras'] == 12
    with pytest.raises(ValueError): scale.estimate(cams[:3], {c['name']: p for c, p in zip(cams, pos)})  # too few photos with GPS
    # DJI XMP (as written by the aircraft) and plain EXIF GPS
    (tmp_path / 'dji.jpg').write_bytes(b'\xff\xd8 <x drone-dji:GpsLatitude="+53.538060825" drone-dji:GpsLongitude="-2.302693401" '
                                        b'drone-dji:AbsoluteAltitude="+189.824" drone-dji:RelativeAltitude="+34.020" drone-dji:RtkFlag="50"/> \xff\xd9')
    assert scale.read_position(tmp_path / 'dji.jpg') == {'lat': 53.538060825, 'lon': -2.302693401, 'alt': 189.824, 'rtk': True}
    from PIL import Image
    ex = Image.Exif(); ex[0x8825] = {1: 'S', 2: (33.0, 52.0, 4.2), 3: 'E', 4: (151.0, 12.0, 30.0), 5: 0, 6: 41.5}
    Image.new('RGB', (8, 8)).save(tmp_path / 'exif.jpg', exif=ex)
    p = scale.read_position(tmp_path / 'exif.jpg')
    assert abs(p['lat'] + 33.86783) < 1e-4 and abs(p['lon'] - 151.20833) < 1e-4 and abs(p['alt'] - 41.5) < 1e-6 and not p['rtk']
    assert scale.read_position(tmp_path / 'missing.jpg') is None

def test_model_scale_endpoints():
    import json, os
    from pathlib import Path
    from app import reconstruct
    from app.database import SessionLocal
    from app.models import Job, Media, Model3D
    cams, pos = _synthetic_flight(scale_m=4.0, n=8)
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'scale'}).json()
        root = Path(os.environ['UPLOAD_DIR'])
        with SessionLocal() as db:
            ids = []
            for i, p in enumerate(pos):  # photos named so their build order is the camera order
                key = f'scale_{ins["id"]}_{i}.jpg'
                (root / key).write_bytes(f'<x drone-dji:GpsLatitude="{p["lat"]}" drone-dji:GpsLongitude="{p["lon"]}" drone-dji:AbsoluteAltitude="{p["alt"]}" drone-dji:RtkFlag="50"/>'.encode())
                m = Media(inspection_id=ins['id'], filename=f'DJI_{i:04d}.JPG', storage_key=key, content_type='image/jpeg', sha256='0' * 64, size_bytes=1); db.add(m); db.flush(); ids.append(m.id)
            job = Job(inspection_id=ins['id'], kind='reconstruction', status='completed', stage='done', media_ids=ids); db.add(job); db.flush()
            model = Model3D(inspection_id=ins['id'], job_id=job.id, dense=False, status='completed', quality={'files': {}}); db.add(model); db.commit(); mid = model.id
            web = reconstruct.RECON_DIR / f'{ins["id"]}_{job.id}' / 'web'; web.mkdir(parents=True); (web / 'cameras.json').write_text(json.dumps(cams))
        assert client.get(f'/api/v1/models/{mid}/scale').json() is None
        s = client.post(f'/api/v1/models/{mid}/scale/detect').json()
        assert abs(s['m_per_unit'] - 4.0) < .01 and s['method'] == 'gps_rtk'
        assert client.get(f'/api/v1/models/{mid}/scale').json()['m_per_unit'] == s['m_per_unit']   # saved with the model
        r = client.put(f'/api/v1/models/{mid}/scale', json={'m_per_unit': 4.1, 'reference_m': 2.6, 'reference_units': .634, 'note': 'antenna'})
        assert r.status_code == 200 and r.json()['method'] == 'manual' and r.json()['m_per_unit'] == 4.1
        assert client.put(f'/api/v1/models/{mid}/scale', json={'m_per_unit': 0}).status_code == 422
        assert client.delete(f'/api/v1/models/{mid}/scale').status_code == 204 and client.get(f'/api/v1/models/{mid}/scale').json() is None
        assert client.post('/api/v1/models/999999/scale/detect').status_code == 404

def test_session_cookie_for_files_and_reads_only(monkeypatch):
    """Images and 3D files are fetched by the browser without the Authorization header: the login also sets an
    HttpOnly cookie that works for reads only, and data files are no longer public."""
    import os
    from pathlib import Path
    monkeypatch.setenv('AUTH_DISABLED', 'false')
    (Path(os.environ['UPLOAD_DIR']) / 'cookie-check.txt').write_text('private')
    with TestClient(app) as client:
        anon = TestClient(app)
        assert anon.get('/uploads/cookie-check.txt').status_code == 401        # data files need a login now
        r = client.post('/api/v1/auth/login', json={'email': 'admin@aeromind.local', 'password': 'AeroMind@123'})
        assert r.status_code == 200 and 'aeromind_session' in r.cookies and 'httponly' in r.headers['set-cookie'].lower()
        assert client.get('/uploads/cookie-check.txt').text == 'private'      # the cookie alone opens files
        assert client.get('/api/v1/sites').status_code == 200                  # and reads
        assert client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'csrf'}).status_code == 401  # but never changes
        h = {'Authorization': 'Bearer ' + r.json()['token']}
        assert client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'header'}, headers=h).status_code == 201
        assert client.post('/api/v1/auth/logout', headers=h).status_code == 204
        assert client.get('/uploads/cookie-check.txt').status_code == 401

def _write_las(path, xyz, crs=None, rgb=None):
    import laspy, numpy as np
    h = laspy.LasHeader(point_format=7 if rgb is not None else 6, version='1.4'); h.scales = [.001] * 3; h.offsets = np.floor(xyz.min(0))
    if crs is not None: h.add_crs(crs)
    las = laspy.LasData(h); las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None: las.red, las.green, las.blue = (rgb.astype('uint16') * 257).T
    las.write(path)

def test_pointcloud_reading_units_and_crop(tmp_path):
    pytest.importorskip('laspy'); np = pytest.importorskip('numpy')
    from pyproj import CRS
    from app import pointcloud as pc
    pts = np.c_[np.linspace(0, 1000, 5000), np.linspace(0, 500, 5000), np.full(5000, 100.0)] + [6_224_000, 2_001_000, 0]
    _write_las(tmp_path / 'feet.laz', pts, CRS.from_epsg(2229))  # NAD83 / California zone 5 (US survey feet)
    d = pc.read_clouds([tmp_path / 'feet.laz'], max_points=10_000)
    assert d['total'] == 5000 and 'foot' in d['units'].lower()
    ft = 1200 / 3937  # US survey foot in metres
    assert abs(np.ptp(d['points'][:, 0]) - 1000 * ft) < .01 and abs(d['points'][0, 2] - 100 * ft) < .01  # converted to metres
    x0 = 6_224_000 * ft; roi = (x0, x0 + 100, -1e12, 1e12)  # crop: first 100 m only
    assert 1500 < pc.read_clouds([tmp_path / 'feet.laz'], max_points=10_000, roi=roi)['kept'] < 1800  # 100 m of a 305 m line
    assert pc.peek_crs([tmp_path / 'feet.laz']).to_epsg() == 2229
    (tmp_path / 'a.xyz').write_text('# x y z r g b\n' + '\n'.join(f'{i} {2*i} {3} 255 0 0' for i in range(10)))
    t = pc.read_clouds([tmp_path / 'a.xyz']); assert t['kept'] == 10 and tuple(t['colors'][0]) == (255, 0, 0)
    with pytest.raises(pc.CloudError): pc.read_clouds([tmp_path / 'a.lax'])

def test_pointcloud_alignment_recovers_pose_and_height_datum():
    np = pytest.importorskip('numpy'); pytest.importorskip('scipy')
    import math
    from app import pointcloud as pc
    rng = np.random.default_rng(5)
    # a "tower": vertical poles and platforms (surfaces the sparse points lie on), in local metres
    poles = np.vstack([np.c_[np.full(3000, x) + rng.normal(0, .02, 3000), np.full(3000, y) + rng.normal(0, .02, 3000), rng.uniform(0, 30, 3000)] for x, y in ((0, 0), (3, 0), (0, 3), (3, 3))])
    deck = np.vstack([np.c_[rng.uniform(-1, 4, 4000), rng.uniform(-1, 4, 4000), np.full(4000, z)] for z in (10, 20, 30)])
    ground = np.c_[rng.uniform(-30, 30, 20000), rng.uniform(-30, 30, 20000), rng.normal(0, .03, 20000)]
    cloud = np.vstack([poles, deck, ground])
    # the camera model's frame: arbitrary scale, rotation and offset; GPS heights 40 m above the cloud's datum
    s, a = 0.25, .8; R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    to_model = lambda X: (X @ R) / s + [5, -2, 1]
    cams = np.c_[20 * np.cos(np.linspace(0, 6, 30)), 20 * np.sin(np.linspace(0, 6, 30)), np.full(30, 25.0)]
    gps = cams + [0, 0, 40] + rng.normal(0, .02, (30, 3))
    sparse = to_model(cloud[rng.choice(len(cloud), 3000, replace=False)] + rng.normal(0, .01, (3000, 3)))
    r = pc.align(to_model(cams), gps, sparse, cloud)
    back = r['s'] * to_model(cams) @ r['R'].T + r['t']
    assert abs(r['vertical_shift_m'] + 40) < .1 and np.abs(back - cams).max() < .1 and r['inliers_pct'] > 80

def test_pointcloud_upload_and_cloud_only_model(monkeypatch):
    pytest.importorskip('laspy'); np = pytest.importorskip('numpy')
    import io, os, tempfile
    from pathlib import Path
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'scan only'}).json()
        assert client.post(f"/api/v1/inspections/{ins['id']}/pointclouds", files={'file': ('x.exe', b'MZ', 'application/octet-stream')}).status_code == 415
        assert client.post(f"/api/v1/inspections/{ins['id']}/pointcloud-model").status_code == 409  # nothing uploaded yet
        tmp = Path(tempfile.mkdtemp()) / 'tower.las'
        z = np.linspace(0, 30, 4000); _write_las(tmp, np.c_[np.cos(z) * 2, np.sin(z) * 2, z] + [500_000, 200_000, 50])
        up = client.post(f"/api/v1/inspections/{ins['id']}/pointclouds", files={'file': ('tower.las', tmp.read_bytes(), 'application/octet-stream')})
        assert up.status_code == 201 and up.json()['file_format'] == 'las' and up.json()['role'] == 'cloud'
        assert client.post(f"/api/v1/inspections/{ins['id']}/pointclouds", files={'file': ('tower.lax', b'LASX', 'application/octet-stream')}).json()['role'] == 'sidecar'
        job = client.post(f"/api/v1/inspections/{ins['id']}/pointcloud-model")
        assert job.status_code == 202, job.text
        m = client.get(f"/api/v1/inspections/{ins['id']}/models").json()[0]
        assert m['status'] == 'completed', m['quality'] and m['quality']['source'] == 'pointcloud'
        q = m['quality']; assert q['boxes_available'] is False and 'no photos' in q['boxes_note'] and q['pointcloud']['points_total'] == 4000
        assert abs(q['scale']['m_per_unit'] * 10 - 30.0) < 3  # ~10 viewer units across a ~30 m tall cloud
        assert client.get(m['urls']['web_points']).status_code == 200
        assert client.delete(f"/api/v1/pointclouds/{up.json()['id']}").status_code == 204

def test_pointcloud_e57_scans_in_global_frame(tmp_path):
    pye57 = pytest.importorskip('pye57'); np = pytest.importorskip('numpy')
    from app import pointcloud as pc
    n = 2000; rng = np.random.default_rng(4)
    with pye57.E57(str(tmp_path / 'scan.e57'), mode='w') as f:
        for shift in (0.0, 100.0):  # two scans; the second placed 100 m east by its pose
            f.write_scan_raw({'cartesianX': rng.uniform(0, 10, n), 'cartesianY': rng.uniform(0, 10, n), 'cartesianZ': rng.uniform(0, 30, n),
                              'intensity': rng.uniform(0, 1, n).astype(np.float32)}, name=f's{shift}', translation=np.array([shift, 0.0, 0.0]), rotation=np.array([1.0, 0, 0, 0]))
    d = pc.read_clouds([tmp_path / 'scan.e57'], max_points=10_000)
    assert d['total'] == 2 * n and d['colors'].shape == (2 * n, 3)
    x = d['points'][:, 0]; assert x.min() >= -.01 and 99 < x.max() <= 110.01  # both scans, poses applied

class _FakeS3:
    """Just enough of boto3's S3 client for app/storage.py."""
    def __init__(self): self.o = {}
    def _miss(self):
        from botocore.exceptions import ClientError
        return ClientError({'Error': {'Code': '404'}}, 'op')
    def head_object(self, Bucket, Key):
        if Key not in self.o: raise self._miss()
        return {'ContentLength': len(self.o[Key])}
    def upload_file(self, f, Bucket, Key): self.o[Key] = open(f, 'rb').read()
    def download_file(self, Bucket, Key, f):
        if Key not in self.o: raise self._miss()
        open(f, 'wb').write(self.o[Key])
    def copy(self, src, Bucket, Key): self.o[Key] = self.o[src['Key']]
    def delete_object(self, Bucket, Key): self.o.pop(Key, None)
    def get_object(self, Bucket, Key, Range=None):
        import io
        if Key not in self.o: raise self._miss()
        data = self.o[Key]; extra = {}
        if Range:
            a, b = Range[6:].split('-'); a, b = int(a), int(b or len(data) - 1); extra = {'ContentRange': f'bytes {a}-{b}/{len(data)}'}; data = data[a:b + 1]
        class Body(io.BytesIO):
            def iter_chunks(self, n): yield self.read()
        return {'Body': Body(data), 'ContentLength': len(data), **extra}
    def create_multipart_upload(self, Bucket, Key): self.mp = getattr(self, 'mp', {}); self.mp['u1'] = {}; return {'UploadId': 'u1'}
    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body): self.mp[UploadId][PartNumber] = Body; return {'ETag': f'"e{PartNumber}"'}
    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        assert [x['ETag'] for x in MultipartUpload['Parts']] == [f'"e{n}"' for n in sorted(self.mp[UploadId])]
        self.o[Key] = b''.join(v for _, v in sorted(self.mp.pop(UploadId).items()))
    def abort_multipart_upload(self, Bucket, Key, UploadId): self.mp.pop(UploadId, None)
    def get_paginator(self, name):
        o = self.o
        class P:
            def paginate(self, Bucket, Prefix): yield {'Contents': [{'Key': k, 'Size': len(v)} for k, v in o.items() if k.startswith(Prefix)]}
        return P()

def test_storage_namespace_bucket_mode_stream_and_trim(monkeypatch, tmp_path):
    import os, time
    from app import storage
    fake = _FakeS3(); root = tmp_path / 'uploads'; root.mkdir()
    monkeypatch.setattr(storage, '_client', fake); monkeypatch.setattr(storage, 'BUCKET', 'b'); monkeypatch.setattr(storage, 'ACCESS_KEY', 'k'); monkeypatch.setattr(storage, 'SECRET_KEY', 's')
    monkeypatch.setattr(storage, '_roots', {'uploads': root.resolve()}); monkeypatch.setattr(storage, 'PREFIX', 'dev/'); monkeypatch.setattr(storage, 'MODE', 'bucket')
    fake.o['uploads/shared.jpg'] = b'production'                          # written by production
    assert storage.fetch(root / 'shared.jpg').read_bytes() == b'production'  # readable from the namespaced copy
    (root / 'new.jpg').write_bytes(b'0123456789'); storage.push(root / 'new.jpg')
    assert 'dev/uploads/new.jpg' in fake.o and 'uploads/new.jpg' not in fake.o  # own writes stay in the namespace
    storage.release(root / 'new.jpg'); assert not (root / 'new.jpg').exists()      # bucket mode: no local copy kept
    body, status, headers = storage.stream(root / 'new.jpg', 'bytes=2-5')
    assert status == 206 and b''.join(body) == b'2345' and headers['Content-Range'] == 'bytes 2-5/10'
    assert storage.stream(root / 'missing.jpg') is None
    storage.delete(root / 'shared.jpg'); assert 'uploads/shared.jpg' in fake.o     # a namespaced copy never deletes production keys
    assert storage.copy(root / 'shared.jpg', root / 'sites' / 'moved.jpg') and fake.o['dev/uploads/sites/moved.jpg'] == b'production'
    # trim: least recently used first, never recent files or folders in use, never the only copy
    old = time.time() - 3 * 3600
    for name in ('a.bin', 'b.bin'):
        p = root / name; p.write_bytes(b'x' * 1000); storage.push(p); os.utime(p, (old, old))
    busy = root / 'build'; busy.mkdir(); (busy / 'big.bin').write_bytes(b'x' * 5000); os.utime(busy / 'big.bin', (old, old))
    (root / 'local-only.bin').write_bytes(b'y' * 1000); os.utime(root / 'local-only.bin', (old, old))
    with storage.in_use(busy):
        freed = storage.trim_cache(max_bytes=1500)  # in-use folders do not count
    assert (busy / 'big.bin').exists() and freed >= 1000
    assert fake.o.get('dev/uploads/local-only.bin') == b'y' * 1000 or (root / 'local-only.bin').exists()  # pushed before removal

def test_cancel_upload_deletes_unprocessed_media_only():
    import os
    from pathlib import Path
    from app.database import SessionLocal
    from app.models import Detection
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'cancel'}).json()
        up = client.post(f"/api/v1/inspections/{ins['id']}/media", files={'file': ('a.jpg', b'\xff\xd8\xff\xd9', 'image/jpeg')}).json()
        path = Path(os.environ['UPLOAD_DIR']) / up['storage_key']; assert path.is_file()
        assert client.delete(f"/api/v1/media/{up['id']}").status_code == 204 and not path.exists()
        assert client.get(f"/api/v1/inspections/{ins['id']}/media").json() == []
        assert client.delete(f"/api/v1/media/{up['id']}").status_code == 404
        done = client.post(f"/api/v1/inspections/{ins['id']}/media", files={'file': ('b.jpg', b'\xff\xd8\xff\xd9', 'image/jpeg')}).json()
        from app.models import Job
        with SessionLocal() as db:
            j = Job(inspection_id=ins['id'], kind='detection', status='completed', stage='detection'); db.add(j); db.flush()
            db.add(Detection(job_id=j.id, inspection_id=ins['id'], media_id=done['id'], frame_index=0, class_id=0, class_name='rru', confidence=.9,
                             x1=0, y1=0, x2=1, y2=1, image_width=2, image_height=2, model_name='t')); db.commit()
        r = client.delete(f"/api/v1/media/{done['id']}"); assert r.status_code == 409 and 'detections' in r.json()['detail']  # processed: kept

def test_chunked_upload_local_and_bucket(monkeypatch):
    """Large files arrive in parts (each request far below the proxy's 300 s limit); parts may come in any order."""
    from app import main, storage
    from app.tasks import ROOT
    monkeypatch.setattr(main, 'UPLOAD_PART', 4)
    data = b'0123456789'  # 3 parts: 4 + 4 + 2 bytes
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'chunked'}).json()
        start = lambda **k: client.post(f"/api/v1/inspections/{ins['id']}/chunked-uploads", json={'kind': 'pointcloud', 'filename': 'scan.laz', 'size': len(data), **k})
        assert start(filename='x.exe').status_code == 415 and start(kind='media', content_type='text/html').status_code == 415
        s = start().json(); assert s['parts'] == 3 and s['part_size'] == 4
        put = lambda uid, n, b: client.put(f'/api/v1/chunked-uploads/{uid}/parts/{n}', content=b)
        assert put(s['upload_id'], 1, b'012').status_code == 400               # wrong part length
        assert put(s['upload_id'], 3, b'89').status_code == 200 and put(s['upload_id'], 1, b'0123').status_code == 200
        assert client.post(f"/api/v1/chunked-uploads/{s['upload_id']}/complete").status_code == 409  # part 2 missing
        assert put(s['upload_id'], 2, b'4567').status_code == 200
        x = client.post(f"/api/v1/chunked-uploads/{s['upload_id']}/complete").json()
        assert x['role'] == 'cloud' and x['size_bytes'] == 10 and (ROOT / x['storage_key']).read_bytes() == data
        assert client.post(f"/api/v1/chunked-uploads/{s['upload_id']}/complete").status_code == 404  # session is gone
        # cancel removes the partial file
        c = start().json(); put(c['upload_id'], 1, b'0123'); key = main._chunked[c['upload_id']]['key']
        assert client.delete(f"/api/v1/chunked-uploads/{c['upload_id']}").status_code == 204 and not (ROOT / key).exists()
        # with a bucket: S3 multipart upload, a video becomes a Media row
        fake = _FakeS3()
        monkeypatch.setattr(storage, '_client', fake); monkeypatch.setattr(storage, 'BUCKET', 'b'); monkeypatch.setattr(storage, 'ACCESS_KEY', 'k'); monkeypatch.setattr(storage, 'SECRET_KEY', 's')
        monkeypatch.setattr(storage, '_roots', {'uploads': ROOT.resolve()}); monkeypatch.setattr(storage, 'PREFIX', '')
        v = start(kind='media', filename='flight.mp4', content_type='video/mp4').json()
        for n, b in ((2, b'4567'), (1, b'0123'), (3, b'89')): assert put(v['upload_id'], n, b).status_code == 200
        m = client.post(f"/api/v1/chunked-uploads/{v['upload_id']}/complete").json()
        assert m['content_type'] == 'video/mp4' and fake.o['uploads/' + m['storage_key']] == data

def test_file_sizes_over_2_gb_fit():
    from sqlalchemy import BigInteger
    from app.models import Media, PointCloudFile
    assert isinstance(Media.__table__.c.size_bytes.type, BigInteger) and isinstance(PointCloudFile.__table__.c.size_bytes.type, BigInteger)
    from sqlalchemy.dialects import postgresql
    assert PointCloudFile.__table__.c.size_bytes.type.compile(dialect=postgresql.dialect()) == 'BIGINT'

def test_cancel_queued_job_only(monkeypatch):
    """A second click queues a duplicate job: it can be cancelled while it waits; a job the worker already took cannot."""
    from app import gpu
    from app.database import SessionLocal
    from app.models import WorkerTask, Job, Model3D
    monkeypatch.setattr(gpu, 'REMOTE', True)
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'twice'}).json()
        with SessionLocal() as db:
            jobs = [Job(inspection_id=ins['id'], status='queued', stage='reconstruction', progress=0, detail='Queued') for _ in range(2)]
            db.add_all(jobs); db.flush()
            for j in jobs: db.add(Model3D(inspection_id=ins['id'], job_id=j.id, status='queued'))
            tasks = [WorkerTask(kind='run_pointcloud_model', lane='long', payload={'job_id': j.id, 'model_id': 0}) for j in jobs]
            db.add_all(tasks); db.flush(); tasks[0].status = 'running'; jobs[0].status = 'running'; db.commit()
            ids, tids = [j.id for j in jobs], [t.id for t in tasks]
        assert client.post(f'/api/v1/jobs/{ids[0]}/cancel').status_code == 409          # running on the GPU
        r = client.post(f'/api/v1/jobs/{ids[1]}/cancel'); assert r.status_code == 200 and r.json()['status'] == 'cancelled'
        assert client.post(f'/api/v1/jobs/{ids[1]}/cancel').status_code == 409          # nothing left to cancel
        assert client.post('/api/v1/jobs/999999/cancel').status_code == 404
        with SessionLocal() as db:
            assert db.get(WorkerTask, tids[1]).status == 'failed' and db.get(WorkerTask, tids[0]).status == 'running'
            assert db.query(Model3D).filter_by(job_id=ids[1]).one().status == 'cancelled'

def test_build_photos_spread_over_the_whole_flight(monkeypatch):
    """A long flight is not cut to its first RECON_MAX_IMAGES photos: the build takes them evenly from start to end."""
    from types import SimpleNamespace
    from app import main, reconstruct
    monkeypatch.setattr(reconstruct, 'MAX_IMAGES', 5)
    photos = [SimpleNamespace(id=i, filename=f'DJI_{i:04d}.JPG') for i in range(100, 0, -1)]  # any upload order
    assert [m.id for m in main._flight_photos(photos)] == [1, 26, 51, 75, 100]
    assert [m.id for m in main._flight_photos(photos[-3:])] == [1, 2, 3]  # few photos: all, in flight order

def _dji_head(lat, lon, alt, rel, yaw, pitch, roll=0.0, w=4000, h=3000, f=3000.0):
    """First bytes of a DJI photo: XMP (RTK position, gimbal angles, lens calibration) and the JPEG frame header."""
    xmp = (f'<x:xmpmeta><rdf:Description drone-dji:GpsLatitude="{lat}" drone-dji:GpsLongitude="{lon}" drone-dji:AbsoluteAltitude="{alt}" '
           f'drone-dji:RelativeAltitude="{rel}" drone-dji:RtkFlag="50" drone-dji:GimbalYawDegree="{yaw}" drone-dji:GimbalPitchDegree="{pitch}" '
           f'drone-dji:GimbalRollDegree="{roll}" drone-dji:DewarpData="2022-01-01;{f},{f},0,0,0,0,0,0,0"/></x:xmpmeta>').encode()
    app1 = b'http://ns.adobe.com/xap/1.0/\x00' + xmp
    sof = bytes([8]) + h.to_bytes(2, 'big') + w.to_bytes(2, 'big') + bytes([3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1])
    return b'\xff\xd8\xff\xe1' + (len(app1) + 2).to_bytes(2, 'big') + app1 + b'\xff\xc0' + (len(sof) + 2).to_bytes(2, 'big') + sof

def test_photo_pose_from_drone_metadata():
    """A camera's metadata pose: the point it aims at projects to the image centre, a point to its right right of it."""
    np = pytest.importorskip('numpy'); pytest.importorskip('pyproj')
    from pyproj import CRS
    from app import photo_pose as pp
    m = pp.read(_dji_head(53.5, -3.0, 150.0, 30.0, yaw=90.0, pitch=-30.0))  # facing east, 30 degrees down, on UTM 30N's central meridian
    assert (m['w'], m['h'], m['fx'], m['cx'], m['rtk']) == (4000, 3000, 3000.0, 2000.0, True)
    cams, info = pp.poses([m], CRS.from_epsg(32630), np.zeros(3)); c = cams[0]
    d = np.array([np.cos(np.radians(30)), 0, -np.sin(np.radians(30))])  # east and down
    uv, z = pp.project(c, np.array([c['C'] + 20 * d, c['C'] + 20 * d + [0, -1, 0], c['C'] + 20 * d + [0, 0, 1]]))
    assert np.allclose(uv[0], [2000, 1500], atol=1) and z[0] > 0
    assert uv[1][0] > 2000 + 100   # one metre south = to the camera's right
    assert uv[2][1] < 1500 - 100   # one metre up = higher in the image
    assert pp.read(b'\xff\xd8 no metadata') is None
    T = pp.target([{'C': np.array([10.0, 0, 5]), 'R': np.vstack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])},
                   {'C': np.array([0.0, 10, 5]), 'R': np.vstack([[1, 0, 0], [0, 0, -1], [0, -1, 0]])}])
    assert np.allclose(T[:2], [0, 0], atol=1e-6)  # two cameras looking at the origin from the east and the north

def test_las_header_crs_resolves_to_epsg_for_the_national_grids():
    """LAS WKT1 'OSGB36 / BNG + ODN height' with a geoid EXTENSION is not matched by PROJ on its own (it then used a
    ballpark transformation: 49 m off in height); it must resolve to EPSG:27700+5701 = 7405."""
    pytest.importorskip('pyproj')
    from pyproj import CRS
    from app import photo_pose as pp
    wkt = ('COMPD_CS["OSGB36 / British National Grid + ODN height",PROJCS["OSGB36 / British National Grid",GEOGCS["OSGB36",DATUM["Ordnance_Survey_of_Great_Britain_1936",'
           'SPHEROID["Airy 1830",6377563.396,299.3249646,AUTHORITY["EPSG","7001"]],AUTHORITY["EPSG","6277"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
           'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4277"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",49],'
           'PARAMETER["central_meridian",-2],PARAMETER["scale_factor",0.9996012717],PARAMETER["false_easting",400000],PARAMETER["false_northing",-100000],'
           'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","27700"]],VERT_CS["ODN height",'
           'VERT_DATUM["Ordnance Datum Newlyn",2005,EXTENSION["PROJ4_GRIDS","uk_os_OSGM15_GB.tif"],AUTHORITY["EPSG","5101"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
           'AXIS["Gravity-related height",UP],AUTHORITY["EPSG","5701"]]]')
    assert pp.authority_crs(CRS.from_wkt(wkt)).to_epsg() == 7405
    assert pp.has_vertical(CRS.from_wkt(wkt)) and not pp.has_vertical(CRS.from_epsg(32630))

def test_pointcloud_model_uses_photo_metadata_not_colmap(monkeypatch, tmp_path):
    """Photos + LAS: cameras from metadata only (COLMAP is never called, no photo is downloaded); the model gets a
    camera model for the 3D-box step, and heights in a CRS without a vertical datum are fixed from the take-off height."""
    np = pytest.importorskip('numpy'); pytest.importorskip('laspy'); pytest.importorskip('scipy')
    from pyproj import CRS, Transformer
    from app import tasks, reconstruct, photo_pose as pp
    monkeypatch.setattr(reconstruct, 'RECON_DIR', tmp_path / 'recon')
    monkeypatch.setattr(reconstruct, 'colmap', lambda *a, **k: (_ for _ in ()).throw(AssertionError('COLMAP must not run')))
    crs = CRS.from_epsg(32630); lat0, lon0 = 53.5, -3.0
    E0, N0 = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon0, lat0)
    rng = np.random.default_rng(1); ground = np.c_[rng.uniform(-40, 40, (20000, 2)), np.zeros(20000)]
    a = rng.uniform(0, 2 * np.pi, 30000); tower = np.c_[1.5 * np.cos(a), 1.5 * np.sin(a), rng.uniform(0, 30, 30000)]
    root = tasks.ROOT / 'test_meta_pose'; (root / 'photos').mkdir(parents=True, exist_ok=True)
    try:
        _write_las(root / 'site.las', np.r_[ground, tower] + [E0, N0, 0], crs=crs)
        to_ll = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True); keys = []
        for i, ang in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):  # an orbit at 20 m, 20 m up, looking at the tower
            x, y = E0 + 20 * np.cos(ang), N0 + 20 * np.sin(ang); lon, lat = to_ll.transform(x, y)
            yaw = np.degrees(np.arctan2(-np.cos(ang), -np.sin(ang)))  # towards the tower (bearing from north)
            p = root / 'photos' / f'DJI_{i:04d}.JPG'; p.write_bytes(_dji_head(lat, lon, 520.0, 20.0, yaw, -20.0)); keys.append(p.relative_to(tasks.ROOT).as_posix())
        r = tasks.pointcloud_model({'work': 'meta_test', 'clouds': [(root / 'site.las').relative_to(tasks.ROOT).as_posix()], 'photos': keys}, lambda *a: None)
        cams = r['pointcloud']['cameras']
        assert r['boxes_available'] and r['camera_source'] == 'metadata' and r['registered_images'] == 12, r
        assert abs(cams['height_corrected_m'] + 500) < 2  # 520 m ellipsoidal take-off height vs ground at 0: corrected by -500
        assert cams['on_cloud_pct'] >= 90, cams
        model = reconstruct.RECON_DIR / 'meta_test' / 'sparse' / '0'
        assert (model / 'images.txt').is_file() and (model / 'cameras.txt').read_text().split()[1] == 'OPENCV'
        assert len(__import__('json').loads((reconstruct.RECON_DIR / 'meta_test' / 'web' / 'cameras.json').read_text())) == 12
    finally:
        import shutil; shutil.rmtree(root, ignore_errors=True)

def test_remote_file_reads_ranges_across_blocks(monkeypatch, tmp_path):
    """Large files are read from the bucket by range (no local copy): seeks and reads across block boundaries."""
    from app import storage
    fake = _FakeS3(); root = tmp_path / 'uploads'; root.mkdir()
    monkeypatch.setattr(storage, '_client', fake); monkeypatch.setattr(storage, 'BUCKET', 'b'); monkeypatch.setattr(storage, 'ACCESS_KEY', 'k'); monkeypatch.setattr(storage, 'SECRET_KEY', 's')
    monkeypatch.setattr(storage, '_roots', {'uploads': root.resolve()}); monkeypatch.setattr(storage, 'PREFIX', ''); monkeypatch.setattr(storage.RemoteFile, 'BLOCK', 10)
    data = bytes(range(256)) * 4; fake.o['uploads/big.las'] = data
    f = storage.open_remote(root / 'big.las')
    assert f.read(25) == data[:25] and f.read(30) == data[25:55]
    f.seek(1000); assert f.read(100) == data[1000:]
    f.seek(7); assert f.read(4) == data[7:11]; f.close()
    assert storage.read_head(root / 'big.las', 16) == data[:16] and storage.read_bytes(root / 'big.las') == data
    assert not (root / 'big.las').exists()  # nothing was saved locally
    assert storage.open_remote(root / 'missing.las') is None

def test_vote_boxes_one_per_object_any_confidence():
    """Point voting (point cloud models): an RRU and an antenna seen by 8 photos give exactly one box each, from
    low-confidence detections too, with the class that covers their points."""
    np = pytest.importorskip('numpy'); pytest.importorskip('scipy')
    from types import SimpleNamespace as NS
    from app import registration as R
    rng = np.random.default_rng(0)
    def block(c, size, n): return c + (rng.random((n, 3)) - .5) * size
    rru, ant = block(np.array([0, 0, 0.0]), [.4, .4, .5], 3000), block(np.array([1.5, 0, .3]), [.4, .15, 1.6], 6000)
    P = np.r_[rru, ant]; W, H, f = 2000, 1500, 1500.0; cam = ('PINHOLE', [f, f, W / 2, H / 2], W, H)
    poses, photos, by_media, names = {}, [], {}, {}
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        C = np.array([.75 + 8 * np.cos(a), 8 * np.sin(a), 1.0]); fwd = np.array([.75, 0, 0]) - C; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); down = np.cross(fwd, right); Rw = np.vstack([right, down, fwd])
        name = f'photos/{i:05d}.jpg'; poses[name] = (None, None, C, 1, Rw, -Rw @ C); m = NS(id=i + 1); photos.append(m); names[m.id] = name
        dets = []
        for k, (pts, cls) in enumerate(((rru, 'rru'), (ant, 'sector_antenna'))):
            uv, _ = R.project(pts, Rw, -Rw @ C, cam); (x1, y1), (x2, y2) = uv.min(0), uv.max(0)
            dets.append(NS(id=i * 10 + k, media_id=m.id, class_name=cls, confidence=.12, x1=x1, y1=y1, x2=x2, y2=y2, image_width=W, image_height=H))
        by_media[m.id] = dets
    Mz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])  # model Z up -> viewer Y up, as in the app
    out = R.vote_candidates(P, Mz, photos, names, by_media, poses, {1: cam})
    assert sorted(c['class_name'] for c in out) == ['rru', 'sector_antenna'], [(c['class_name'], c['point_count']) for c in out]
    a = next(c for c in out if c['class_name'] == 'sector_antenna')
    assert abs(a['size_h'] - 1.6) < .3 and np.allclose(a['position'], [1.5, .3, 0], atol=.15)
    assert all(c['media_count'] >= 4 and len(c['detection_ids']) >= 4 for c in out)

def test_tower_split_from_trees_and_clear_start_view():
    """Unclassified clouds: grey tower points stay the structure, green trees around it go to the surroundings layer,
    and the viewer starts on the side with no trees between the camera and the tower."""
    np = pytest.importorskip('numpy'); pytest.importorskip('scipy')
    from app import pointcloud as pc
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 20000); tower = np.c_[1.0 * np.cos(a), 1.0 * np.sin(a), rng.uniform(0, 20, 20000)]
    top = np.c_[rng.uniform(-2.5, 2.5, (8000, 2)), rng.uniform(16, 20, 8000)]  # antennas reaching 2.5 m out
    trees = np.c_[rng.uniform(-12, 12, 30000), rng.uniform(-12, -3, 30000), rng.uniform(0, 9, 30000)]  # forest on the south side
    P = np.r_[tower, top, trees]; C = np.r_[np.full((28000, 3), 150), np.tile([60, 140, 50], (30000, 1))].astype(np.uint8)
    S, info = pc.split_structure(P, C, np.ones(len(P), bool))
    assert S[:28000].mean() > .95 and S[28000:].mean() < .02, (S[:28000].mean(), S[28000:].mean())
    assert 2.5 <= info['reach_m'] <= 4.0
    M, _ = pc.viewer_transform(P)
    view = pc.clear_view(P, S, info, M)
    cam = ((np.asarray(view['position']) - M[:3, 3]) @ (M[:3, :3] / np.linalg.norm(M[:3, 0]))) / np.linalg.norm(M[:3, 0])  # back to metres
    assert cam[1] > -1 and np.hypot(cam[0], cam[1]) > 20  # not on the south side, where all the trees are, and far enough out

def test_las_upload_is_converted_to_copc_and_streamed_by_range():
    """Upload a LAS: the GPU worker converts it to COPC with Untwine (like the PLA QC app); the browser gets a signed
    link that serves byte ranges without the sign-in header (copc.js / Potree) and nothing else."""
    np = pytest.importorskip('numpy'); pytest.importorskip('laspy')
    from app import copc
    if not copc.untwine_exe(): pytest.skip('untwine not installed')
    import tempfile
    from pathlib import Path
    with TestClient(app) as client:
        ins = client.post('/api/v1/inspections', json={'site_id': 1, 'label': 'copc'}).json()
        tmp = Path(tempfile.mkdtemp()) / 'site.las'; rng = np.random.default_rng(0)
        _write_las(tmp, rng.uniform(0, 50, (20000, 3)) + [400000, 300000, 10], crs=__import__('pyproj').CRS.from_epsg(32630), rgb=rng.integers(0, 255, (20000, 3)))
        up = client.post(f"/api/v1/inspections/{ins['id']}/pointclouds", files={'file': ('site.las', tmp.read_bytes(), 'application/octet-stream')}).json()
        f = next(x for x in client.get(f"/api/v1/inspections/{ins['id']}/pointclouds").json() if x['id'] == up['id'])
        assert f['copc_status'] == 'ready' and f['copc_points'] == 20000 and f['copc_key'].endswith('site.copc.laz'), f
        url = f['copc_url']; assert '.copc.laz' in url  # Potree picks its COPC loader from the name
        anon = TestClient(app)
        r = anon.get(url, headers={'Range': 'bytes=0-374'}); assert r.status_code == 206 and r.content[:4] == b'LASF' and len(r.content) == 375
        parts = url.split('/'); parts[6] = '0' * 32
        assert anon.get('/'.join(parts)).status_code == 403  # tampered signature
        parts = url.split('/'); parts[5] = '1000'
        assert anon.get('/'.join(parts)).status_code == 403  # expired
        assert client.post(f"/api/v1/pointclouds/{up['id']}/convert").status_code == 202  # rebuild on request
        assert client.delete(f"/api/v1/pointclouds/{up['id']}").status_code == 204
