﻿using DwgDump.Db;
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
		private readonly string dataDir = @"C:\git\dwg-crawl\Data";

		private readonly string fullPath;
		private readonly string fileId;

		private readonly Document teighaDocument;
		public DbMongo db;

		public CrawlAcDbDocument(CrawlDocument crawlDoc)
		{
			this.fullPath = crawlDoc.Path;
			this.fileId = crawlDoc.FileId;
			this.teighaDocument = TeighaApp.DocumentManager.Open(Path.Combine(dataDir, crawlDoc.FileId + ".dwg"));
		}

		public void DumpDocument()
		{
			// If document wasn't loaded correctly
			if (this.teighaDocument == null)
				return;

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
						this.db.SaveObjectData(objectJson, this.fileId);
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

					crawlAcDbLayerTableRecord cltr = new crawlAcDbLayerTableRecord(layerTblRec);
					string objectJson = JsonHelper.To<crawlAcDbLayerTableRecord>(cltr);

					this.db.SaveObjectData(objectJson, this.fileId);
				}

				// Run all xefs
				List<CrawlDocument> xrefs = GetXrefs(this.teighaDocument);
				foreach (CrawlDocument theXref in xrefs)
				{
					CrawlAcDbDocument cDoc = new CrawlAcDbDocument(theXref);
					db.InsertIntoFiles(theXref);

					cDoc.db = db;
					cDoc.DumpDocument();
				}
			}
			this.teighaDocument.CloseAndDiscard();

			db.SetDocumentScanned(this.fileId);
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

				this.db.InsertIntoFiles(blockJson);

				using (Transaction tr = aDoc.TransactionManager.StartTransaction())
				{
					foreach (ObjectId obj in btr)
					{
						string objectJson = DumpEntity2json(obj);
						// string objectClass = obj.ObjectClass.Name;

						this.db.SaveObjectData(objectJson, cBtr.BlockId);
					}
				}
			}
			else if (objId.ObjectClass.Name == "AcDbProxyEntity")
			{
				Entity ent = (Entity)objId.GetObject(OpenMode.ForRead);
				DBObjectCollection dbo = new DBObjectCollection();
				ent.Explode(dbo);

				crawlAcDbProxyEntity cPxy = new crawlAcDbProxyEntity((ProxyEntity)ent)
				{
					BlockId = Guid.NewGuid().ToString(),
					FileId = parentFileId
				};

				string pxyJson = JsonHelper.To<crawlAcDbProxyEntity>(cPxy);

				this.db.InsertIntoFiles(pxyJson);

				foreach (ObjectId obj in dbo)
				{
					string objectJson = DumpEntity2json(obj);
					// string objectClass = obj.ObjectClass.Name;

					this.db.SaveObjectData(objectJson, cPxy.BlockId);
				}
			}
		}

		private string DumpEntity2json(ObjectId id_platf)
		{
			string result = "";

			try
			{
				// Всякое может случиться
				// Открываем переданный в функцию объект на чтение, преобразуем его к Entity
				Entity ent = (Entity)id_platf.GetObject(OpenMode.ForRead);

				//Далее последовательно проверяем класс объекта на соответствие классам основных примитивов

				if (id_platf.ObjectClass.Name == "AcDbLine")
				{
					// Если объект - отрезок (line)
					crawlAcDbLine kline = new crawlAcDbLine((Line)ent); //Преобразуем к типу линия
					result = JsonHelper.To<crawlAcDbLine>(kline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbPolyline")
				{
					// Если объект - полилиния
					Polyline kpLine = (Polyline)ent;
					crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
					result = JsonHelper.To<crawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDb2dPolyline")
				{
					// 2D полилиния - такие тоже попадаются
					Polyline2d kpLine = (Polyline2d)ent;
					crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
					result = JsonHelper.To<crawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDb3dPolyline")
				{
					// 2D полилиния - такие тоже попадаются
					Polyline3d kpLine = (Polyline3d)ent;

					crawlAcDbPolyline jpline = new crawlAcDbPolyline(kpLine);
					result = JsonHelper.To<crawlAcDbPolyline>(jpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbText")
				{
					// Текст
					DBText dbtxt = (DBText)ent;
					crawlAcDbText jtext = new crawlAcDbText(dbtxt);
					result = JsonHelper.To<crawlAcDbText>(jtext);
				}
				else if (id_platf.ObjectClass.Name == "AcDbMText")
				{
					// Мтекст
					MText mtxt = (MText)ent;
					crawlAcDbMText jtext = new crawlAcDbMText(mtxt);
					result = JsonHelper.To<crawlAcDbMText>(jtext);
				}
				else if (id_platf.ObjectClass.Name == "AcDbArc")
				{
					// Дуга
					Arc arc = (Arc)ent;
					crawlAcDbArc cArc = new crawlAcDbArc(arc);
					result = JsonHelper.To<crawlAcDbArc>(cArc);
				}
				else if (id_platf.ObjectClass.Name == "AcDbCircle")
				{
					// Окружность
					Circle circle = (Circle)ent;
					crawlAcDbCircle cCircle = new crawlAcDbCircle(circle);
					result = JsonHelper.To<crawlAcDbCircle>(cCircle);
				}
				else if (id_platf.ObjectClass.Name == "AcDbEllipse")
				{
					// Эллипс
					Ellipse el = (Ellipse)ent;
					crawlAcDbEllipse cEll = new crawlAcDbEllipse(el);
					result = JsonHelper.To<crawlAcDbEllipse>(cEll);
				}
				else if (id_platf.ObjectClass.Name == "AcDbAlignedDimension")
				{
					// Размер повернутый
					AlignedDimension dim = (AlignedDimension)ent;

					crawlAcDbAlignedDimension rDim = new crawlAcDbAlignedDimension(dim);
					result = JsonHelper.To<crawlAcDbAlignedDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbRotatedDimension")
				{
					// Размер повернутый
					RotatedDimension dim = (RotatedDimension)ent;

					crawlAcDbRotatedDimension rDim = new crawlAcDbRotatedDimension(dim);
					result = JsonHelper.To<crawlAcDbRotatedDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbPoint3AngularDimension")
				{
					// Угловой размер по 3 точкам
					Point3AngularDimension dim = (Point3AngularDimension)ent;

					crawlAcDbPoint3AngularDimension rDim = new crawlAcDbPoint3AngularDimension(dim);
					result = JsonHelper.To<crawlAcDbPoint3AngularDimension>(rDim);
				}

				else if (id_platf.ObjectClass.Name == "AcDbLineAngularDimension2")
				{//Еще угловой размер по точкам
					LineAngularDimension2 dim = (LineAngularDimension2)ent;

					crawlAcDbLineAngularDimension2 rDim = new crawlAcDbLineAngularDimension2(dim);
					result = JsonHelper.To<crawlAcDbLineAngularDimension2>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbDiametricDimension")
				{
					// Размер диаметра окружности
					DiametricDimension dim = (DiametricDimension)ent;
					crawlAcDbDiametricDimension rDim = new crawlAcDbDiametricDimension(dim);
					result = JsonHelper.To<crawlAcDbDiametricDimension>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbArcDimension")
				{
					// Дуговой размер
					ArcDimension dim = (ArcDimension)ent;
					crawlAcDbArcDimension rDim = new crawlAcDbArcDimension(dim);
					result = JsonHelper.To<crawlAcDbArcDimension>(rDim);

				}
				else if (id_platf.ObjectClass.Name == "AcDbRadialDimension")
				{
					// Радиальный размер
					RadialDimension dim = (RadialDimension)ent;
					crawlAcDbRadialDimension rDim = new crawlAcDbRadialDimension(dim);
					result = JsonHelper.To<crawlAcDbRadialDimension>(rDim);
				}
				else if (id_platf.ObjectClass.Name == "AcDbAttributeDefinition")
				{
					// Атрибут блока
					AttributeDefinition ad = (AttributeDefinition)ent;

					crawlAcDbAttributeDefinition atd = new crawlAcDbAttributeDefinition(ad);
					result = JsonHelper.To<crawlAcDbAttributeDefinition>(atd);
				}
				else if (id_platf.ObjectClass.Name == "AcDbHatch")
				{
					// Штриховка
					Teigha.DatabaseServices.Hatch htch = ent as Teigha.DatabaseServices.Hatch;

					crawlAcDbHatch cHtch = new crawlAcDbHatch(htch);
					result = JsonHelper.To<crawlAcDbHatch>(cHtch);
				}
				else if (id_platf.ObjectClass.Name == "AcDbSpline")
				{
					// Сплайн
					Spline spl = ent as Spline;

					crawlAcDbSpline cScpline = new crawlAcDbSpline(spl);
					result = JsonHelper.To<crawlAcDbSpline>(cScpline);
				}
				else if (id_platf.ObjectClass.Name == "AcDbPoint")
				{
					// Точка
					DBPoint Pnt = ent as DBPoint;
					crawlAcDbPoint pt = new crawlAcDbPoint(Pnt);
					result = JsonHelper.To<crawlAcDbPoint>(pt);
				}

				else if (id_platf.ObjectClass.Name == "AcDbBlockReference")
				{
					// Блок
					BlockReference blk = ent as BlockReference;
					crawlAcDbBlockReference cBlk = new crawlAcDbBlockReference(blk);

					result = JsonHelper.To<crawlAcDbBlockReference>(cBlk);

					//newDocument(id_platf, result);
				}
				else if (id_platf.ObjectClass.Name == "AcDbProxyEntity")
				{
					// Прокси
					ProxyEntity pxy = ent as ProxyEntity;

					crawlAcDbProxyEntity cBlk = new crawlAcDbProxyEntity(pxy);

					result = JsonHelper.To<crawlAcDbProxyEntity>(cBlk);

					DocumentFromBlockOrProxy(id_platf, result);
				}
				else if (id_platf.ObjectClass.Name == "AcDbSolid")
				{
					// Солид 2Д
					Solid solid = (Solid)ent;


					crawlAcDbSolid cSld = new crawlAcDbSolid(solid);

					result = JsonHelper.To<crawlAcDbSolid>(cSld);

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

					CrawlDebug.WriteLine("Не могу обработать тип объекта: " + id_platf.ObjectClass.Name);
				}
			}
			catch (System.Exception ex)
			{
				// Если что-то сломалось, то в командную строку выводится ошибка
				CrawlDebug.WriteLine("Не могу преобразовать - ошибка: " + ex.Message);
			};

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
