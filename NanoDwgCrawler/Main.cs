using DwgDump.Data;
using DwgDump.Enitites;
using DwgDump.MultiContent;
using Multicad;
using Multicad.AplicationServices;
using Multicad.DatabaseServices;
using Multicad.Geometry;
using Multicad.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace DwgDump
{
	public static class Main
	{
		static CrawlAcDbDocument currentDocument = null;

		[CommandMethod("sfd", CommandFlags.NoCheck | CommandFlags.NoPrefix)]
		public static void ScanFolder()
		{
			// DirectoryScanner.Scan(@"C:\Users\mihan\Desktop\DwgCrawlTest");
			DirectoryScanner.Scan(@"C:\py\Data");
		}

		/// <summary>
		/// Opens unscanned document
		/// </summary>
		[CommandMethod("grd", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void GetRandomUnscannedDocument()
		{
			//While Get random dwg from database that not scanned
			var doc = DbMongo.Instance.GetNewRandomUnscannedDocument();

			if (doc != null)
			{
				// You will only see it once
				DbMongo.Instance.SetDocumentScanned(doc.FileId);

				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();
			}
		}

		/// <summary>
		/// Writes selection as a fragment to database
		/// </summary>
		[CommandMethod("fr", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Redraw)]
		public static void WriteAndHighligt()
		{
			WriteFragment(method: "highlight");
		}

		[CommandMethod("frd", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Redraw)]
		public static void WriteAndDelete()
		{
			WriteFragment(method: "delete");
		}

		public static void WriteFragment(string method = "highlight")
		{
			List<McObjectId> currentSelection = new List<McObjectId>(McObjectManager.SelectionSet.CurrentSelection);
			if (currentSelection.Count == 0)
				currentSelection = McObjectManager.SelectObjects("Select fragment").ToList();

			// каждый набор объектов со своим Guid группы
			var groupId = Guid.NewGuid().ToString();
			var fileId = currentDocument?.FileId;
			CrawlAcDbDocument.DumpFragment(currentSelection, fileId, groupId);

			// Отфильтровываем объекты аннотации
			var annotationIds = new List<McObjectId>();
			var graphicsIds = new List<McObjectId>();
			foreach (var id in currentSelection)
				if (Converters.IsAnnotation(id))
					annotationIds.Add(id);
				else
					graphicsIds.Add(id);

			// Create preview from everything to get bound
			(Bitmap imageAnnotated, BoundBlock boundAnnotated) = BmpFromDwg.CreatePreview(selectedIds: currentSelection);

			// Create preview from grapics only
			(Bitmap strippedImage, BoundBlock boundStripped) = BmpFromDwg.CreatePreview(selectedIds: graphicsIds, bound: boundAnnotated);

			McContext.ShowNotification(string.Format("graphic entries: {0} annotation entries {1}", graphicsIds.Count, annotationIds.Count));

			CrawlAcDbDocument.DumpFragmentDescription(fileId, groupId, imageAnnotated, boundAnnotated, strippedImage);

			foreach (var id in currentSelection)
			{
				var dbo = id.GetObject()?.Cast<McEntity>()?.DbEntity;
				// Visualise processed entities with color on the drawing
				if (dbo != null)
				{
					if (method == "highlight")
						dbo.Color = Color.DarkSeaGreen;
					if (method == "delete")
						dbo.Erase();
				}
			}
		}

		/// <summary>
		/// Rescans saved fragments to update objects' fileds in database
		/// </summary>
		[CommandMethod("rescan", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void Rescan()
		{
			foreach (CrawlDocument doc in DbMongo.Instance.GetAllScannedDocuments())
			{
				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();

				foreach (long handle in DbMongo.Instance.GetHandlesFromDoc(doc.FileId))
				{
					McObjectId id = McObjectId.FromHandle(handle);
					if (id.IsNull)
						continue;

					var obj = id.GetObject();
					if (obj != null)
					{
						var cent = Converters.From(id);

						DbMongo.Instance.UpdateObject(handle, Converters.Serialize(cent));
					}
				}

				currentDocument.Document.Close();
			}
		}

		[CommandMethod("ccs", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void CloseAll()
		{
			foreach (McDocument doc in McDocumentsManager.GetDocuments())
			{
				doc.Close();
			}
		}


	}
}
