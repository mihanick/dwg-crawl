using DwgDump.Data;
using DwgDump.Enitites;
using DwgDump.Util;
using HostMgd.ApplicationServices;
using HostMgd.EditorInput;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Teigha.DatabaseServices;
using TeighaApp = HostMgd.ApplicationServices.Application;

namespace DwgDump
{
	class CrawlAcDbDocument
	{
		private readonly string fullPath;
		private readonly string fileId;

		private readonly Document teighaDocument;

		private DbMongo Db => DbMongo.Instance;

		// https://stackoverflow.com/questions/2987559/check-if-a-file-is-open
		private bool IsFileinUse(FileInfo file)
		{
			FileStream stream = null;

			try
			{
				stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None);
			}
			catch (IOException)
			{
				//the file is unavailable because it is:
				//still being written to
				//or being processed by another thread
				//or does not exist (has already been processed)
				return true;
			}
			finally
			{
				if (stream != null)
					stream.Close();
			}

			return false;
		}

		public CrawlAcDbDocument(CrawlDocument crawlDoc)
		{
			this.fullPath = crawlDoc.Path;
			this.fileId = crawlDoc.FileId;
			string path = Path.Combine(DbMongo.Instance.DataDir, crawlDoc.FileId + ".dwg");
			FileInfo info = new FileInfo(path);
			if (!IsFileinUse(info))
				this.teighaDocument = TeighaApp.DocumentManager.Open(path);
		}

		public void DumpDocument()
		{
			// If document wasn't loaded correctly
			if (this.teighaDocument == null)
				return;

			// nanoCAD can crash, or exception or whatever...
			// so there will be only one try per document
			// so we first set document scanned, 
			// than try to process it
			Db.SetDocumentScanned(this.fileId);
			try
			{
				using (Transaction tr = this.teighaDocument.TransactionManager.StartTransaction())
				{
					PromptSelectionResult r = this.teighaDocument.Editor.SelectAll();

					// for all entities in drawing
					foreach (SelectedObject obj in r.Value)
					{
						string objId = obj.ObjectId.ToString();
						string objectJson = DumpEntity2json(obj.ObjectId);
						string objectClass = obj.ObjectId.ObjectClass.Name;

						if (!string.IsNullOrEmpty(objectJson))
							this.Db.SaveObjectData(objectJson, this.fileId);
					}

					// also run all blocks
					List<ObjectId> blocks = GetBlocks(this.teighaDocument);
					foreach (ObjectId btrId in blocks)
					{
						BlockTableRecord btr = (BlockTableRecord)btrId.GetObject(OpenMode.ForRead);
						DocumentFromBlockOrProxy(btrId, this.fileId);
					}

					// all layer definitions
					// http://forums.autodesk.com/t5/net/how-to-get-all-names-of-layers-in-a-drawing-by-traversal-layers/td-p/3371751
					LayerTable lt = (LayerTable)this.teighaDocument.Database.LayerTableId.GetObject(OpenMode.ForRead);
					foreach (ObjectId ltr in lt)
					{
						string objId = ltr.ToString();
						string objectClass = ltr.ObjectClass.Name;
						LayerTableRecord layerTblRec = (LayerTableRecord)ltr.GetObject(OpenMode.ForRead);

						CrawlAcDbLayerTableRecord cltr = new CrawlAcDbLayerTableRecord(layerTblRec);
						string objectJson = JsonHelper.To<CrawlAcDbLayerTableRecord>(cltr);

						this.Db.SaveObjectData(objectJson, this.fileId);
					}

					// Run all xefs
					List<CrawlDocument> xrefs = GetXrefs(this.teighaDocument);
					foreach (CrawlDocument theXref in xrefs)
					{
						CrawlAcDbDocument cDoc = new CrawlAcDbDocument(theXref);
						Db.InsertIntoFiles(theXref);
						cDoc.DumpDocument();
					}
				}
				this.teighaDocument.CloseAndDiscard();
			}
			catch (Exception er)
			{
				CrawlDebug.WriteLine(er.Message);
			}
		}

		private void DocumentFromBlockOrProxy(ObjectId objId, string parentFileId)
		{
			//http://www.theswamp.org/index.php?topic=37860.0
			Document aDoc = Application.DocumentManager.GetDocument(objId.Database);

			if (objId.ObjectClass.Name == "AcDbBlockTableRecord")
			{
				BlockTableRecord btr = (BlockTableRecord)objId.GetObject(OpenMode.ForRead);
				CrawlAcDbBlockTableRecord cBtr = new CrawlAcDbBlockTableRecord(btr, this.fullPath)
				{
					BlockId = Guid.NewGuid().ToString(),
					FileId = parentFileId
				};

				string blockJson = JsonHelper.To<CrawlAcDbBlockTableRecord>(cBtr);

				this.Db.InsertIntoFiles(blockJson);

				using (Transaction tr = aDoc.TransactionManager.StartTransaction())
				{
					foreach (ObjectId obj in btr)
					{
						string objectJson = DumpEntity2json(obj);
						// string objectClass = obj.ObjectClass.Name;

						this.Db.SaveObjectData(objectJson, cBtr.BlockId);
					}
				}
			}
			else if (objId.ObjectClass.Name == "AcDbProxyEntity")
			{
				Entity ent = (Entity)objId.GetObject(OpenMode.ForRead);
				using (DBObjectCollection dbo = new DBObjectCollection())
				{
					ent.Explode(dbo);

					CrawlAcDbProxyEntity cPxy = new CrawlAcDbProxyEntity((ProxyEntity)ent)
					{
						BlockId = Guid.NewGuid().ToString(),
						FileId = parentFileId
					};

					string pxyJson = JsonHelper.To<CrawlAcDbProxyEntity>(cPxy);

					this.Db.InsertIntoFiles(pxyJson);

					for (int i = 0; i < dbo.Count; i++)
					{
						DBObject obj = dbo[i];

						if (obj == null)
							continue;

						// BUG: Exploded ids will be null
						string objectJson = DumpEntity2json(obj.Id);

						this.Db.SaveObjectData(objectJson, cPxy.BlockId);
					}
				}
			}
		}

		private string DumpEntity2json(ObjectId id_platf)
		{
			string result = "";

			if (id_platf.IsNull)
				return result;

			try
			{
				// Всякое может случиться
				// Открываем переданный в функцию объект на чтение, преобразуем его к Entity
				Entity ent = (Entity)id_platf.GetObject(OpenMode.ForRead);

				//Далее последовательно проверяем класс объекта на соответствие классам основных примитивов

				if (id_platf.ObjectClass.Name == "AcDbLine")
				{
					// Если объект - отрезок (line)
					CrawlAcDbLine kline = new CrawlAcDbLine((Line)ent); //Преобразуем к типу линия
					result = JsonHelper.To<CrawlAcDbLine>(kline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbPolyline")
				{
					// Если объект - полилиния
					Polyline kpLine = (Polyline)ent;
					CrawlAcDbPolyline jpline = new CrawlAcDbPolyline(kpLine);
					result = JsonHelper.To<CrawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDb2dPolyline")
				{
					// 2D полилиния - такие тоже попадаются
					Polyline2d kpLine = (Polyline2d)ent;
					CrawlAcDbPolyline jpline = new CrawlAcDbPolyline(kpLine);
					result = JsonHelper.To<CrawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDb3dPolyline")
				{
					// 2D полилиния - такие тоже попадаются
					Polyline3d kpLine = (Polyline3d)ent;

					CrawlAcDbPolyline jpline = new CrawlAcDbPolyline(kpLine);
					result = JsonHelper.To<CrawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbText")
				{
					// Текст
					DBText dbtxt = (DBText)ent;
					CrawlAcDbText jtext = new CrawlAcDbText(dbtxt);
					result = JsonHelper.To<CrawlAcDbText>(jtext);
				}
				else if (id_platf.ObjectClass.Name == "AcDbMText")
				{
					// Мтекст
					MText mtxt = (MText)ent;
					CrawlAcDbMText jtext = new CrawlAcDbMText(mtxt);
					result = JsonHelper.To<CrawlAcDbMText>(jtext);
				}
				else if (id_platf.ObjectClass.Name == "AcDbArc")
				{
					// Дуга
					Arc arc = (Arc)ent;
					CrawlAcDbArc cArc = new CrawlAcDbArc(arc);
					result = JsonHelper.To<CrawlAcDbArc>(cArc);
				}
				else if (id_platf.ObjectClass.Name == "AcDbCircle")
				{
					// Окружность
					Circle circle = (Circle)ent;
					CrawlAcDbCircle cCircle = new CrawlAcDbCircle(circle);
					result = JsonHelper.To<CrawlAcDbCircle>(cCircle);
				}
				else if (id_platf.ObjectClass.Name == "AcDbEllipse")
				{
					// Эллипс
					Ellipse el = (Ellipse)ent;
					CrawlAcDbEllipse cEll = new CrawlAcDbEllipse(el);
					result = JsonHelper.To<CrawlAcDbEllipse>(cEll);
				}
				else if (id_platf.ObjectClass.Name == "AcDbAlignedDimension")
				{
					// Размер повернутый
					AlignedDimension dim = (AlignedDimension)ent;

					CrawlAcDbAlignedDimension rDim = new CrawlAcDbAlignedDimension(dim);
					result = JsonHelper.To<CrawlAcDbAlignedDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbRotatedDimension")
				{
					// Размер повернутый
					RotatedDimension dim = (RotatedDimension)ent;

					CrawlAcDbRotatedDimension rDim = new CrawlAcDbRotatedDimension(dim);
					result = JsonHelper.To<CrawlAcDbRotatedDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbPoint3AngularDimension")
				{
					// Угловой размер по 3 точкам
					Point3AngularDimension dim = (Point3AngularDimension)ent;

					CrawlAcDbPoint3AngularDimension rDim = new CrawlAcDbPoint3AngularDimension(dim);
					result = JsonHelper.To<CrawlAcDbPoint3AngularDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbLineAngularDimension2")
				{//Еще угловой размер по точкам
					LineAngularDimension2 dim = (LineAngularDimension2)ent;

					CrawlAcDbLineAngularDimension2 rDim = new CrawlAcDbLineAngularDimension2(dim);
					result = JsonHelper.To<CrawlAcDbLineAngularDimension2>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbDiametricDimension")
				{
					// Размер диаметра окружности
					DiametricDimension dim = (DiametricDimension)ent;
					CrawlAcDbDiametricDimension rDim = new CrawlAcDbDiametricDimension(dim);
					result = JsonHelper.To<CrawlAcDbDiametricDimension>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbArcDimension")
				{
					// Дуговой размер
					ArcDimension dim = (ArcDimension)ent;
					CrawlAcDbArcDimension rDim = new CrawlAcDbArcDimension(dim);
					result = JsonHelper.To<CrawlAcDbArcDimension>(rDim);

				}
				else if (id_platf.ObjectClass.Name == "AcDbRadialDimension")
				{
					// Радиальный размер
					RadialDimension dim = (RadialDimension)ent;
					CrawlAcDbRadialDimension rDim = new CrawlAcDbRadialDimension(dim);
					result = JsonHelper.To<CrawlAcDbRadialDimension>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbAttributeDefinition")
				{
					// Атрибут блока
					AttributeDefinition ad = (AttributeDefinition)ent;

					CrawlAcDbAttributeDefinition atd = new CrawlAcDbAttributeDefinition(ad);
					result = JsonHelper.To<CrawlAcDbAttributeDefinition>(atd);
				}
				else if (id_platf.ObjectClass.Name == "AcDbHatch")
				{
					// Штриховка
					Teigha.DatabaseServices.Hatch htch = ent as Teigha.DatabaseServices.Hatch;

					CrawlAcDbHatch cHtch = new CrawlAcDbHatch(htch);
					result = JsonHelper.To<CrawlAcDbHatch>(cHtch);
				}
				else if (id_platf.ObjectClass.Name == "AcDbSpline")
				{
					// Сплайн
					Spline spl = ent as Spline;

					CrawlAcDbSpline cScpline = new CrawlAcDbSpline(spl);
					result = JsonHelper.To<CrawlAcDbSpline>(cScpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbPoint")
				{
					// Точка
					DBPoint Pnt = ent as DBPoint;
					CrawlAcDbPoint pt = new CrawlAcDbPoint(Pnt);
					result = JsonHelper.To<CrawlAcDbPoint>(pt);
				}

				else if (id_platf.ObjectClass.Name == "AcDbBlockReference")
				{
					// Блок
					BlockReference blk = ent as BlockReference;
					CrawlAcDbBlockReference cBlk = new CrawlAcDbBlockReference(blk);

					result = JsonHelper.To<CrawlAcDbBlockReference>(cBlk);

					//newDocument(id_platf, result);
				}
				else if (id_platf.ObjectClass.Name == "AcDbProxyEntity")
				{
					// Прокси
					ProxyEntity pxy = ent as ProxyEntity;

					CrawlAcDbProxyEntity cBlk = new CrawlAcDbProxyEntity(pxy);

					result = JsonHelper.To<CrawlAcDbProxyEntity>(cBlk);

					DocumentFromBlockOrProxy(id_platf, result);
				}
				else if (id_platf.ObjectClass.Name == "AcDbSolid")
				{
					// Солид 2Д
					Solid solid = (Solid)ent;


					CrawlAcDbSolid cSld = new CrawlAcDbSolid(solid);

					result = JsonHelper.To<CrawlAcDbSolid>(cSld);

				}
				/*
		else if (id_platf.ObjectClass.Name == "AcDbLeader")
		{  //Выноска Autocad
			Leader ld = (Leader)ent;

			if (ld.EndPoint.Z != 0 || ld.StartPoint.Z != 0)
			{
				//ed.WriteMessage("DEBUG: Преобразован объект: Выноска Autocad");

				ld.EndPoint = new Point3d(ld.EndPoint.X, ld.EndPoint.Y, 0);
				ld.StartPoint = new Point3d(ld.StartPoint.X, ld.StartPoint.Y, 0);

				result = true;
			};

		}
		/*
	else if (id_platf.ObjectClass.Name == "AcDbPolygonMesh")
	{
		 BUG: В платформе нет API для доступа к вершинам сетей AcDbPolygonMesh и AcDbPolygonMesh и AcDbSurface

	}
	else if (id_platf.ObjectClass.Name == "AcDbSolid")
	{
		 BUG: Чтобы плющить Solid-ы нужны API функции 3d
	}
	else if (id_platf.ObjectClass.Name == "AcDbRegion")
	{
		Region rgn = ent as Region;
		BUG: нет свойств у региона
	}

	*/
				else
				{
					// Если объект не входит в число перечисленных типов,
					// то выводим в командную строку класс этого необработанного объекта

					CrawlDebug.WriteLine("[TODO] Не могу обработать тип объекта: " + id_platf.ObjectClass.Name);
				}
			}
			catch (System.Exception ex)
			{
				// Если что-то сломалось, то в командную строку выводится ошибка
				CrawlDebug.WriteLine("Не могу преобразовать - ошибка: " + ex.Message);
			}

			// Возвращаем значение функции
			return result;
		}

		/// <summary>
		/// Функция возвращает список блоков с их атрибутами
		/// </summary>
		/// <param name="aDoc"></param>
		/// <returns></returns>
		private List<ObjectId> GetBlocks(Document aDoc)
		{
			Database aDocDatabase = aDoc.Database;


			// Находим таблицу описаний блоков 
			BlockTable blkTbl = (BlockTable)aDocDatabase.BlockTableId
				.GetObject(OpenMode.ForRead, false, true);

			// Открываем таблицу записей текущего чертежа
			BlockTableRecord bt =
				(BlockTableRecord)aDocDatabase.CurrentSpaceId
					.GetObject(OpenMode.ForRead);

			// Переменная списка блоков
			List<ObjectId> bNames = new List<ObjectId>();

			// Пример итерации по таблице определений блоков
			// https://sites.google.com/site/bushmansnetlaboratory/sendbox/stati/multipleattsync
			// Как я понимаю, здесь пробегается по всем таблицам записей,
			// в которых определения блоков не являются анонимными
			// и не являются листами
			foreach (BlockTableRecord btr in blkTbl.Cast<ObjectId>().Select(n =>
				(BlockTableRecord)n.GetObject(OpenMode.ForRead, false))
				.Where(n => !n.IsAnonymous && !n.IsLayout))
			{

				bNames.Add(btr.ObjectId);

				btr.Dispose();
			};

			return bNames;
		}


		private List<CrawlDocument> GetXrefs(Document aDoc)
		{
			// http://adndevblog.typepad.com/autocad/2012/06/finding-all-xrefs-in-the-current-database-using-cnet.html
			XrefGraph xGraph = aDoc.Database.GetHostDwgXrefGraph(false);
			int numXrefs = xGraph.NumNodes;
			List<CrawlDocument> result = new List<CrawlDocument>();

			for (int i = 0; i < numXrefs; i++)
			{
				XrefGraphNode xrefNode = xGraph.GetXrefNode(i);

				if (xrefNode.XrefStatus == XrefStatus.Resolved)
				{
					//Document theDoc = TeighaApp.DocumentManager.GetDocument(xrefNode.Database);
					CrawlDocument acDoc = new CrawlDocument(xrefNode.Database.Filename);
					result.Add(acDoc);
				}
			}
			return result;
		}

	}
}
